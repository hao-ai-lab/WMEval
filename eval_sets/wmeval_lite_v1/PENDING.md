# WMEval Lite release gaps — Unity-free, plug-and-play dataset

Updated 2026-10-05 for `lanxiang/release-cleanup`.

Release promise: a new user can download a versioned dataset and evaluator,
generate their model's videos from the supplied conditions, and evaluate them
against baked Unity reference quantities and visual assets **without installing
or running Unity, accessing a private scene repository, or asking maintainers
for missing files**. Unity is an authoring dependency only. Candidate-side CV
models and hosted judges can still require GPUs, downloads, API keys, and cost;
those requirements must be explicit, not confused with Unity-free operation.

Priorities in this file follow the requested release policy:

- **P0 — must be done now:** blockers to that release promise or to trustworthy
  scores for the advertised release scope.
- **P1 — before-release improvements:** robustness, calibration, and usability
  improvements once the P0 contract works.
- **P2 — after-release improvements:** extensions and optimizations that must
  not delay the first reproducible release.

An item is done only when its acceptance evidence is linked here. An old paper
result, a local successful run, or a valid checksum is not by itself evidence
that a fresh downloader can reproduce the released protocol. Historical
priority labels in `PRODUCTION_READINESS.md` and
`experiments/evaluation_dimension_standardization.yaml` have not yet been
synchronized with this reprioritization; use this register for release planning.

## Verified current state and principal gaps

Read-only audit of the current worktree on 2026-10-05:

- `python3 tools/run_wmeval_lite.py validate --suite eval_sets/wmeval_lite_v1`
  passed for **19 case bundles**; `list-cases` reports **8 dimensions** and
  **31 case-dimension assignments**. This validates bundles, not full CV/judge
  execution or public-download reproducibility.
- All **169 files under the case directories are Git-tracked** (43,569,009
  bytes in this checkout). Existing videos, images, Unity arrays, rubrics,
  causal QA, and portable liquid NPZ artifacts are useful release foundations;
  this is not a claim that the reference assets are all missing.
- All **19** case locks have empty `dependency_versions` and
  `artifact_schema_versions`; **12/19** cases use
  `candidate_normalization: none` (3 use `beamdomino_native`, 4 liquid
  normalization). See `cases/*/*/case.yaml` and
  `evaluation/lite/normalization.py`.
- The [HF dataset proposal](HF_DATASET_PROPOSAL.md) is still a draft: its flat
  table does not yet specify the complete numerical-reference, frame-index,
  action-schema, and evaluator contracts. Its release destination/license and
  scope remain TBD. No public downloadable release was verified in this audit.
- The existing
  [Table 4/7 reference package](../../experiments/tables_4_7/REFERENCE_PACKAGE.md)
  is a real build/verify path for **14 active table cells**, not a full-suite
  dataset exporter. It includes additional ignored/private reference assets;
  its runbook explicitly requires distribution clearance before public upload.
- `tools/run_wmeval_lite.py` already supports validation, discovery, strict
  candidate manifests, evaluation, dry runs, and resume. Generation remains
  split among model-specific scripts; a documented model-neutral generation
  adapter and release-wide job manifest are missing. The Lite runner builds
  work from discovered candidates, so absent expected case/repeat rows need a
  separate completeness gate, not just file validation.
- `eval_sets/README.md` and the HF draft have stale scope statements.
  `requirements-lite.txt` is not a full evaluator environment; macOS and CUDA
  locks exist separately. No root license or checked-in `.github` CI workflow
  was found. `control_motion.py` still resolves an external PTLFlow checkout,
  and some CV models are fetched by mutable aliases.

The critical path is: clear distribution rights and freeze scope; export the
complete reference/conditioning contract; make generation and evaluation
portable; prove a clean-download end-to-end run. Do not silently drop difficult
dimensions or equate unsupported model controls with zero benchmark performance.

## Case-data intake procedure

Deliverables for each planned-but-not-yet-bundled case. Case folders are
created **only when the data lands** (the trust layer rejects placeholders:
everything under `cases/` must validate). Wiring steps for every case:

1. `mkdir -p cases/<Scene>/<variant>/reference/artifacts` and drop the files
2. write `case.yaml` (copy a live case of the same kind as template)
3. `python tools/build_lite_case.py cases/<Scene>/<variant>` (checksums + lock)
4. add the case id to the dimension's `cases:` list in `suite.yaml`
5. `python tools/run_wmeval_lite.py validate` (+ `generate-rubrics` where noted)

Mark a backlog item complete with its validation evidence as data lands. See
`suite.yaml`'s dated header banners for the history of earlier work.

The active open-model Table 4/7 execution plan and live completion evidence
are tracked in
[`TABLE4_TABLE7_REMAINING_MODELS_TODO.md`](../../TABLE4_TABLE7_REMAINING_MODELS_TODO.md).
That checklist covers Matrix-Game 2.0, Matrix-Game 3.0, and LingBot-World Base
Cam for the requested paper cells. It does not close the broader P0/P1 release
contracts below; those remain open until their suite-wide evidence lands.

---

## Release cleanup pending register

Owners below are responsibility areas, not newly assigned individuals. A
`multi-model` check must exercise more than one model/control interface.

### P0 — must be done now: release blockers

- [ ] **P0-01 · Release owner — freeze scope and redistribution rights.**
  Declare the 19-case/8-dimension release roster, repeat policy, supported
  metrics, and the relationship to the separate Table 4/7 protocols. Obtain
  clearance for Unity renders, scene-derived quantities, annotations, and any
  redistributed third-party assets; add dataset/code licenses, attribution,
  citation, and a dataset card. Do not interpret the existing private package
  as permission to publish. **Done when:** a versioned release manifest and
  approved distribution inventory cover every shipped file and advertised cell.

- [ ] **P0-02 · Dataset infrastructure — downloadable, immutable full-suite
  package.** Implement the HF/archival export and download/verify path, reusing
  the existing bundle checksums and Table 4/7 packager rather than inventing a
  second incompatible layout. Include `suite.yaml`, case configs, all referenced
  assets, reference frames/quantities, and any required supplemental contracts;
  publish sizes, hashes, dataset revision, evaluator revision, and schema
  versions. No dependency may resolve into a maintainer's ignored `runs/`,
  private repository, or Modal volume. **Done when:** an anonymous fresh download
  to an arbitrary directory validates every released case without missing files.

- [ ] **P0-03 · Dimension owners — complete baked reference contract.**
  Publish a per-case/per-metric inventory with numerical arrays, units, shapes,
  coordinate conventions, object/anchor IDs, validity masks, and timestamps.
  Include the exact conditioning frame and baked evaluator frame sequences or
  lossless frame shards, with frame indices, timebase, sampling/decoding rules,
  and hashes; a video path alone does not specify the comparison frames.
  Inventory poses/intrinsics, object trajectories and masses/diameters,
  mechanical quantities or their deterministic derivation, calibration/anchors,
  control-flow targets, causal events/QA, lighting references where applicable,
  and the existing fluid masks/flow assets. Distinguish direct Unity exports
  from reconstructed calibration and CV/VLM-derived pseudo-labels; reference-free
  metrics must say so rather than invent Unity targets. **Done when:** every
  reference input consumed by each released metric is documented, bundled, and
  loadable without Unity or reference-side annotation services.

- [ ] **P0-04 · Controls + dimension owners — executable generation conditions.**
  Freeze a versioned per-case prompt/first-frame/action manifest: target entity,
  action channels and key order, units, coordinate frame/handedness, absolute
  versus relative poses, timestamps, duration/fps, and required capabilities.
  Bundle the BilliardGame, ChessBoard walk, Garden, and lighting supplemental
  controls; preserve their exact-versus-reconstructed provenance. Resolve the
  two missing `actions.txt` files and the comment-only ChessBoard table through
  executable manifests, not generator-specific guesses. **Done when:** every
  case has a machine-readable schedule and unsupported camera/object-control
  combinations are rejected or explicitly reported before generation.

- [ ] **P0-05 · Generation infrastructure — new-model adapter and job contract.**
  Provide a documented adapter interface/template and a working example that
  consume the release manifest, declare capabilities, and emit candidate
  manifests accepted by the existing Lite runner. Export the complete expected
  case/repeat/seed job list; support batch execution, resume/retry, isolated
  outputs, and generation receipts with model/checkpoint, actual prompt/control
  hashes, input-frame hash, parameters, and output hash. Keep benchmark-specific
  evaluator overrides in the release contract rather than requiring users to
  hand-author them. **Done when:** a new backend can be connected without edits
  to benchmark internals; a fixture adapter proves the plumbing and a real
  supported backend demonstrates generation-to-evaluation. This does not require
  implementing every model or pretending all models support every action type.

- [ ] **P0-06 · Runtime infrastructure — supported install and dependency
  preflight.** Supply a tested full-evaluation environment for at least one
  advertised platform, including ffmpeg and the external PTLFlow synthetic-flow
  interface/calibration. Populate all 19 dependency/schema locks; pin CV code
  and weight revisions/hashes for GroundingDINO, SAM2, depth models, DINOv2,
  RAFT/OpenPIV, and LPIPS, including fluid checkpoints. A preflight must resolve
  these dependencies from documented public downloads or user-supplied caches,
  list GPU/RAM/API requirements, and fail before expensive work. **Done when:**
  a fresh environment installs and passes dependency/artifact checks without
  hard-coded developer paths or silently using mutable model aliases.

- [ ] **P0-07 · Runner + metric owners — normalization and validity contract.**
  Lock fps, duration, time origin, frame alignment, resolution, and crop/pad
  transforms for every scored case. Preserve raw candidates and record the
  applied transform. Standardize per-metric status/reason, units, source,
  coverage/denominator, evaluator fingerprint, and normalization provenance.
  Define anchor visibility/assignment/paired-frame gates for EOC, missing-label
  behavior for mechanics, and missing-event behavior for causal LPIPS. Retain
  valid missing-object penalties while distinguishing measurement failure from
  model failure; unsupported or unmeasurable values must not become favorable
  zeros or silently disappear from aggregate coverage. **Done when:** malformed,
  mismatched, missing-anchor, and unsupported-control fixtures produce explicit,
  tested outcomes; causal scores never silently become temporal fallback scores.

- [ ] **P0-08 · Judge owners — frozen reference judgments and honest API
  reproducibility.** Bake reference causal events, QA/answers, and rubrics keyed
  by case/reference hash, judge identity, prompt version, and exact sampled
  frames. Reuse those fixtures for every candidate; do not re-judge Unity per
  candidate. Record candidate-side judge requests/responses, available snapshot
  metadata, retries, refusals, and repeats; state which metrics require API
  credentials/egress and what is omitted in an API-free run. Do not promise
  bitwise repeatability or immutable snapshots where providers do not offer
  them. **Done when:** reference scoring is fixed, judge-dependent results are
  auditable, and regression tolerances/rebaselining rules are documented.

- [ ] **P0-09 · Results owner — comparable results and completeness checks.**
  Separate generation seeds/repeats from judge repeats and document aggregation
  axes, `ddof`, effective sample sizes, missingness, and capability exclusions.
  Add a gate comparing expected jobs and metric rows with actual outputs; the
  current discovery-driven runner alone cannot detect omitted cases. Bind resume
  and result validity to candidate bytes, parameters, and evaluator identity,
  not just a reused model/case/seed path. Publish a reproducible baseline artifact
  set with per-case results, manifests, receipts, commands, and tolerance bands;
  keep released-protocol replication distinct from exact original-paper parity.
  **Done when:** incomplete runs cannot be presented as complete leaderboards,
  and a downloader can reproduce the shipped baseline's supported metrics.

- [ ] **P0-10 · Release infrastructure — clean-download acceptance gate and
  accurate quickstart.** Test the published package from a clean temporary
  directory/environment with no Unity installation, private checkout, or
  pre-existing developer caches. Exercise download/checksums, preflight, job
  export, adapter fixture, real candidate ingestion, all released dimensions,
  resume, and aggregation; publish explicit coverage and expected outputs.
  Add a lightweight CI gate plus an explicit GPU/API release smoke gate, with
  no hidden paid calls in default tests. Update stale README/HF scope tables,
  commands, and generated readiness priorities. **Done when:** a third party
  follows the documented workflow unassisted and the exact release artifacts
  pass; a local bundle-validation pass alone does not close this item.

Existing implementation to retain: RacingCar first-person is excluded from
active Table 4 scoring; the CUDA handoff is hashed and relocatable; causal and
temporal LPIPS fallbacks are separated; missing required kinematics labels
invalidate FDE/DTW; the suite can retry error rows and invalidate changed
evaluator fingerprints. These are partial foundations, not P0 completion.

### P1 — improvements before release: stronger and more robust benchmark

- [ ] **Controls + model maintainers (`multi-model`) — calibration coverage.**
  Exercise camera-pose, keyboard/mouse, and object-action interfaces; validate
  timing/gain and target success without changing the semantic trajectory.
  Add BeamDomino object-strike and bottle/camera adapters where feasible.
  Unsupported declarations are mandatory in P0; wider adapter support is P1.
- [ ] **Kinematics owner — stronger physical calibration.** Add held-out
  reprojection/self-match checks and near-zero-motion error floors. Quantify
  depth-domain bias and rounding sensitivity; use physically interpretable
  error measures and label exact match as diagnostic where appropriate.
  Explicit units and honest validity are already P0 requirements.
- [ ] **EOC owner (`multi-model`) — anchor/depth robustness.** Calibrate
  visibility and assignment thresholds across Mirror, Billiard, and Garden;
  test reflective surfaces, occlusion, disappearance, and camera-away/return.
  Publish per-scene self-match bounds and the photogrammetric depth caveat.
- [ ] **Action controllability owner — scale and nuisance robustness.** Compare
  pixel EPE and relative EPE, stress-test resolution/camera compensation, and
  validate the synthetic-flow baseline on multiple backends. The dependency
  and frame/coordinate contract must already be portable under P0.
- [ ] **Judge + prompt owners — uncertainty and sensitivity.** Quantify repeated
  judge variance, calibrate against human-reviewed fixtures, and stress-test
  sampling density, prompts, and refusals. Preserve all physical-causality
  judge passes; do not confuse five judge passes with five model generations.
- [ ] **Lighting + fluid owners — dimension-specific rigor.** Recover a more
  exact original fixed-sun controller trace when available, preserving existing
  reconstructed provenance; test lighting estimability and fluid camera-motion
  assumptions. Keep fluid per-profile/multi-axis results canonical and label
  any scalar headline as navigation-only.
- [ ] **Release infrastructure — broader portability and diagnostics.** Add
  more supported OS/GPU combinations, optional containers or project extras,
  adversarial codec/truncation tests, runtime/cost estimates, and actionable
  diagnostics. Expand CI beyond the minimal P0 clean-release gate.
- [ ] **Dataset/results owners — broader baselines and documentation.** Add
  heterogeneous models, negative controls, per-scene failure examples, and
  uncertainty analyses. Provide a worked adapter tutorial and model capability
  matrix; require release-protocol evidence rather than borrowing paper numbers.

### P2 — after-release improvements

- [ ] **Dataset expansion.** Add ChessBoard lighting only after a scope decision;
  add moving-sun lighting only with its own sweep-aware contract. More scenes,
  longer trajectories, harder memory/control cases, and held-out splits should
  have new versioned protocols rather than changing frozen v1 scores.
- [ ] **Performance and packaging.** Add streaming/sharded delivery, partial
  downloads, shared CV feature caches, multi-node evaluation, and richer plugin
  packaging after the basic full download and supported install are reliable.
- [ ] **Optional fully offline judges.** Evaluate local-judge alternatives and
  cross-calibration; keep their scores distinct from the hosted-judge protocol.
  Unity-free v1 must not be mislabeled as requiring zero API access for all metrics.
- [ ] **Evidence lifecycle.** Automate archival, retention, and dry-run cleanup
  of resumable caches and superseded reports without deleting canonical
  releases, baseline evidence, or candidate receipts.
- [ ] **Benchmark maintenance.** Add version migration tools, deprecation policy,
  a submission/leaderboard service, and ongoing drift checks against frozen
  fixtures; publish new versions instead of overwriting released references.

## Release acceptance sequence

1. Close P0-01 and freeze the case/metric roster and versioned schemas.
2. Close P0-02 through P0-04: publish complete, rights-cleared conditioning and
   reference assets, including frames and numerical quantities.
3. Close P0-05 through P0-09: demonstrate adapter-to-candidate-to-evaluation
   interoperability with explicit supported/unsupported coverage and provenance.
4. Close P0-10 against the actual downloadable artifact; record the release
   revision, environment, commands, expected results, and smoke-test evidence.
5. Ship only with every P0 closed. Track P1 improvements before release and P2
   extensions afterwards. Any scope change requires an explicit release decision,
   not a quiet reduction to whichever metrics happen to pass.

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

The historical run counts and observations below are retained from the prior
checklist for dimension owners; the 2026-10-05 audit above did not re-run those
CV/API experiments. Their release implications are prioritized in P0/P1/P2.

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
