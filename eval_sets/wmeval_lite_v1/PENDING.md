# WMEval Lite release gaps — Unity-free, plug-and-play dataset

Updated 2026-10-08 for `lanxiang/release-cleanup` (open-model runners from
`lanxiang/release-cleanup-oss-models` integrated; action-backend registry added;
per-dimension READMEs merged into the root `README.md`).

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
- The HF draft still has stale scope statements (the stale
  `eval_sets/README.md` was folded into the root `README.md` on 2026-10-08).
  `requirements-lite.txt` is not a full evaluator environment; macOS and CUDA
  locks exist separately. No root license or checked-in `.github` CI workflow
  was found. `control_motion.py` still resolves an external PTLFlow checkout,
  and some CV models are fetched by mutable aliases.

Follow-up audit on 2026-10-08 (after integrating the open-model runners):

- `pytest tests` passes (677 passed, 1 skipped) with the ignored `runs/` data
  and the sibling paper checkout present; tests that read `runs/` or
  `../_arxiv__WorldModel_Eval` are not yet hermetic.
- **Table 4/7 action standardization exists; Lite does not.** All five Table
  4/7 backends (Unity, Project Genie, Matrix-Game 2.0/3.0, LingBot Base Cam)
  compile from the hash-locked semantic trajectories. Action targets, native
  timelines, supported/unsupported targets with reasons, and frozen camera
  gains are declared once in
  [`action_backends.yaml`](../../experiments/tables_4_7/action_backends.yaml)
  and enforced by `evaluation/action_backends.py`; input preparation refuses
  unsupported target/backend pairs. The 19 Lite bundles still mix
  `actions.txt` browser programs, unnamed 4/5-column `action.npy` arrays and
  prose, with no target/frame semantics.
- **Lite action/target mismatch.** `RacingCar/prompt_third` is wired to
  `kinematics_mechanics` and `environment_object_consistency`, but its action
  is a camera-only "hold ArrowLeft 10 s" orbit around a parked car, so the
  kinematics row measures a static vehicle. The Table 4 drive trajectory
  (`racing_car_drive.yaml`, steer/throttle/brake) is the correct control.
- **Cross-model calibration is not uniform.** Matrix-Game 2.0 uses one pooled
  gain; Matrix-Game 3.0 uses four regime/direction gains fitted on
  WanderingGarden, which is also a scored scene; LingBot keeps nominal poses
  and realizes -17.7/+41.4 degrees for +/-45-degree perturbations and
  +54.9/+277.9 degrees for full turns. Consistency scores therefore compare
  different viewpoints unless realized yaw is reported alongside them.
- **Repeat policy differs by model.** LingBot runs an adaptive 5-to-30
  expansion (360-row queue); other models use five runs.
- `experiments/tables_4_7/evaluator_assets.lock.json` now pins GroundingDINO,
  Depth Anything (metric outdoor), SAM 2.1 and DINOv2 by revision and hash for
  the Table 4/7 evaluator; RAFT, LPIPS/VGG, PTLFlow, OpenPIV and the indoor
  depth checkpoint are not pinned, and the 19 Lite case locks remain empty.
- Reference-side judgments are not all baked: causal-LPIPS contact events are
  re-judged on the Unity video for every candidate; the Unity-reference
  prompt-consistency scores reported in the paper are not stored in the
  bundles; `lighting_shading` is reference-free and its bundle has no Unity
  video.

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

## Concrete P0 release specification

This is the **target contract to implement**, not a claim that the files or
commands described below already exist. Existing CLI commands are identified
separately. Closing P0 means delivering this end-to-end product, not just more
reference videos. Changes here specify work; they do not launch generations,
authorize additional paid evaluations, or publish data.

### A. What a downloader receives

Keep the existing `suite.yaml` / `cases/<Scene>/<variant>/case.yaml` layout and
extend it with versioned schemas and explicit generation/reference manifests.
The evaluator is a separately pinned code release; it is not a Unity project.
The following filenames are proposed additions except for the existing suite,
case bundles, checksums, and evaluator outputs noted below.

```text
wmeval_lite_v1/
  README.md, DATASET_CARD.md, LICENSE, ATTRIBUTION.md
  release.json                     # dataset/code revisions, scope, file hashes/sizes
  suite.yaml                       # existing case-to-dimension mapping
  schemas/                         # generation, actions, references, jobs, results
  metric_catalog.json              # formulas, units, direction, validity, aggregation
  expected_jobs.jsonl               # complete case/repeat/seed roster, before generation
  evaluator_assets.lock.json        # CV code/weights, schemas, environment identities
  cases/<Scene>/<variant>/
    case.yaml                      # existing evaluator config; no private paths
    generation.json                # prompt, image, action schedule, timing, capabilities
    conditioning/
      first_frame.png              # exact RGB conditioning pixels, not decode-at-runtime
      actions.json                 # typed, timed controls with explicit target entities
      camera.npz                   # ONLY if camera-pose conditioning is required
    reference/
      manifest.json                # per-metric input inventory and provenance
      checksums.sha256             # retain existing checksum/lock machinery
      <existing-reference>.mp4     # where reference-based; useful for visual inspection
      frames/                      # lossless frames consumed by the reference evaluator
      frame_index.json             # frame IDs, timestamps, sampling/alignment, hashes
      artifacts/                   # existing arrays/JSON/NPZ, extended as required
  baselines/<baseline_id>/          # rights-cleared candidates, receipts, expected scores
  acceptance/                      # clean-install evidence, fixtures, tolerance checks
```

Do not duplicate an existing `reference/ref.png` unnecessarily: a manifest may
point to that frozen file. Every one of the 19 cases must resolve to an explicit
conditioning image with a pixel/byte hash; no adapter may independently choose
or extract it. The audit found **10 explicit first frames and 9 video-frame-0
fallbacks**. Bake and verify the latter once, checking that frame time, camera
pose, object state, and the start of the action schedule agree. If an authored
sketch is the intended input, retain and label it; do not quietly replace it
with a Unity frame. An intentional image/reference mismatch needs an explicit
protocol rationale and alignment test.

Ship all reference frames a metric actually reads: metrics that track every
frame need the full evaluation-rate sequence, not just a judge contact sheet.
Lossless shards are an equivalent storage layout. Runtime evaluators must
consume these frozen frames or a verified equivalent decoding path; merely
adding PNGs that the evaluator ignores does not close P0-03. Reference-free
cases need no invented reference movie. Every optional artifact is declared
required, not applicable with a reason, or missing/blocking in the inventory.

All paths inside the dataset must be relative and relocatable. Every file has
a size/hash and provenance, and every array has a schema; no executable pickle
is required to load reference data. Reject escaping paths and unsafe archive
members. A rights-cleared baseline-video archive may be downloaded separately,
but the release manifest must pin it and the acceptance test must retrieve it.
Future rollout frames/answers are **evaluation-only**, never model inputs.

### B. Exactly which Unity references and annotations are needed

The release roster is **19 cases across 9 scenes, 8 dimensions, and 31
case-dimension assignments**, not all 9 × 8 combinations. One case's rollout
is reused by its assigned dimensions. A dimension is a scoring view, not a
request to regenerate the same condition. `suite.yaml` remains authoritative.

| Dimension (case count) | Required baked evaluation inputs, where applicable | Returned measurements |
| --- | --- | --- |
| `lpips` (6) | Exact RGB comparison frames/timebase; for causal LPIPS, frozen reference event IDs, times/windows and labels plus matching policy. Freeze reference-side judgments instead of re-running them per candidate. | Perceptual distance, event-match coverage; causal and temporal-only values separately labeled. |
| `action_controllability` (2) | Timed numeric controls with target object/camera; reference frames; intrinsics/extrinsics and synthetic-flow calibration/targets used by `control_motion`. Bake reference-side flow or supply a pinned, Unity-free derivation. | Motion endpoint error, compensated/dynamic variants and bad-pixel rates, with valid pixel/frame counts. |
| `prompt_scene_consistency` (6) | Canonical prompt and conditioning image; frozen environment/object rubrics, judge sampling and scoring configuration. No Unity trajectory target is required by this reference-free metric. | Environment, object and overall adherence, item-level judgments and coverage. |
| `lighting_shading` (1) | Fixed-sun condition and action protocol, shadow/object anchors, masks/configuration used by the pipeline, and pinned analytic geometry-floor parameters/code or baked outputs. This case has no matching fixed-sun Unity movie; do not substitute the rotating-sun capture. | SRR, SDR, MOSD, SDJ, SAS, SSD, estimability and floor context; not a fabricated Unity-video similarity score. |
| `environment_object_consistency` (6) | Reference RGB frames; camera poses/intrinsics; stable object/character IDs, Unity world transforms, anchor definitions and visibility/validity masks; required reference tracking/depth/appearance caches or pinned derivations. | Object permanence, pairwise-distance/depth errors and character consistency where applicable, with anchor/frame coverage. |
| `kinematics_mechanics` (3) | Same camera/object correspondence data; masses and size/calibration quantities used by the metric; reference 2D/3D trajectories, timebase, velocities/momentum/energy or their exact deterministic derivation and smoothing rule. | FDE/DTW and mechanical error/exact-match outputs where supported, per object, with units and missing-label validity. |
| `physical_causality` (3) | Reference frames; fixed QA, answer keys, event order/contact labels and timing where available; judge rubrics/version and repeated-judgment policy. Distinguish simulation contact records from judged events. | Per-question/per-step causal correctness and aggregate causal scores, judge repeat details and failures. |
| `fluid_mechanics` (4) | Four profiles (rest, step-hold, impulse-decay, chirp-strafe); reference frames/timebase, bottle/control trace, liquid/bottle masks, flow/PIV arrays, surface/volume/spectrum references consumed by the four modules, or pinned Unity-free derivations. Reuse portable NPZ artifacts. | Velocity, volume conservation, surface orientation and slosh-spectrum results per profile; keep axes separate. |

`reference/manifest.json` must list each asset's consuming metric and
`source_kind`: direct Unity export, recovered calibration, analytic derivation,
CV pseudo-label, VLM annotation, or human annotation. Include original source
hash, extraction code/configuration and confidence/validity where relevant.
For example, SAM/RAFT outputs on a Unity-rendered video are **not ground-truth
fluid particle velocities**. Recovered camera poses are not direct Unity poses.
Do not promise unexported dense depth, segmentation, collision impulses, or
particle fields as ground truth; add them only if a released metric needs them
and they can be obtained with verified provenance.

Array contracts must state dtype/shape, seconds/frame correspondence, metres
versus Unity units, axis directions/handedness, quaternion ordering, and whether
poses are camera-to-world or world-to-camera. Intrinsics must specify the image
resolution and pixel convention; masks and projected anchors must use the same
image geometry. Freeze the existing per-case conventions or publish explicit
conversions rather than relabeling old arrays. Acceptance fixtures must check
reprojection, temporal alignment, object-ID joins, array lengths and validity.
Ship measured self-match baselines/tolerances: CV-based evaluation of a Unity
video against itself need not be zero or perfect.

### C. How users generate a new or existing model's rollouts

1. Download a pinned dataset revision and verify every manifest/hash. Choose
   the full release protocol or a named subset; a subset must stay labeled.
2. Export the complete job list **before** checking which candidates exist.
   Each job records `job_id`, dataset/condition hashes, case ID, model identity,
   repeat ID, requested seed, expected output, and assigned dimensions. A
   proposed baseline is five repeats with seeds `42, 1, 2, 3, 4`, matching prior
   Lite runs, **pending P0-01 sign-off**. That would mean **95 generations and
   155 dimension-result rows per fully supported model**, before internal
   judge repeats. Table 4/7's separate repeat protocol is not this default.
3. Connect an adapter with two operations: capability validation and generation.
   Input: job plus `generation.json` and only its allowlisted conditioning
   assets. Output: raw video plus receipt. Ship an adapter template, a no-cost
   deterministic fixture, and one real supported example. Existing models use
   the same contract; no model-specific changes to evaluator internals.
4. The adapter supplies the exact prompt/image and compiles controls to its
   backend. `generation.json` must declare duration, requested timestamps/fps,
   frame count, native/evaluation geometry, perspective, image preprocessing,
   interpolation, prompt policy and required control channels. `actions.json`
   must declare time intervals and sampling/hold rules, target entity, typed
   channel/value/units, key order, coordinate frame, and absolute/relative mode.
   State whether output frame 0 is the supplied image or the first prediction;
   specify action-to-frame latency and the last-frame endpoint convention.
5. Reject or record unsupported conditions **before** spending GPU/API time.
   Keyboard control of the camera is not control of the ball. A text-only model
   cannot silently enter the image-conditioned track; an image-only model must
   not silently drop a required text/control channel. Pose, keyboard and other
   backends may compile the same semantic schedule only with documented
   calibration and fidelity checks. Approximate/prompt-only substitutes must
   be labeled as a separate protocol, not pooled with exact-control results.
6. Write the raw video, timing metadata and a generation receipt: checkpoint
   revision/hash, adapter code, requested/actual seed (including unsupported
   seeds), full parameters, actual prompt/image/control hashes, compiled
   controls, retries, output hash and deviations. Apply the released
   normalization once for evaluation; preserve raw output and transform log.
   Resume only when input/model/config/output identities still match.
7. Emit the existing runner's strict `candidates.jsonl`, joining it to the
   richer receipt/job manifest by identity. Do not add unrecognized fields to
   the current candidate schema. Existing videos may be imported only after
   checking conditions and provenance; otherwise mark them legacy/exploratory.

All 19 cases need a concrete duration/action schedule; **standardized does not
mean forcing every scene to have the same duration or the same trajectory**.
The current action inventory has 15 native machine-readable cases and four
requiring supplemental contracts: BilliardGame/prompt_first,
ChessBoard/walk_third, WanderingGarden/scene_first, SunRisesAndSets/lighting.
Machine-readable does not establish correct action-target semantics or support
on every backend. Resolve these gaps in the release manifests, not ad hoc
adapter overrides.

This is replay of **fixed, open-loop conditions**. Arbitrary interactive actions
cannot be scored against the same baked trajectory as though it were their
Unity counterfactual. New action sequences require newly authored references
or a separately declared reference-free protocol. Adapters must not receive
future reference RGB, object trajectories, reference answers, or metric outputs;
test this input boundary even though the public dataset itself is inspectable.

### D. How the evaluation codebase will run

After installing the pinned evaluator environment and downloading its pinned
CV assets, users supply their candidate manifest. Hosted-judge credentials are
needed only for declared judge-dependent metrics. Dataset reference preparation
must require neither Unity nor reference-side API calls. Candidate-side CV and
judge computation still runs. Preflight must show dimensions, GPU/memory needs,
API egress and estimated cost; a no-API selection is explicitly partial, not
an all-metrics run. Paid tests require an explicit opt-in.

These commands **already exist**; the paths illustrate a downloaded dataset
and candidate manifest, not a currently published release URL:

```bash
python tools/run_wmeval_lite.py validate --suite /data/wmeval_lite_v1
python tools/run_wmeval_lite.py list-cases --suite /data/wmeval_lite_v1
python tools/run_wmeval_lite.py run \
  --suite /data/wmeval_lite_v1 \
  --candidate-manifest /data/my_model/candidates.jsonl \
  --output-dir /data/results/my_model --dry-run
python tools/run_wmeval_lite.py run \
  --suite /data/wmeval_lite_v1 \
  --candidate-manifest /data/my_model/candidates.jsonl \
  --output-dir /data/results/my_model --device cuda --resume
```

An existing-format manifest row is:

```json
{"model":"my_model","case_id":"BallRoller/ctrl","seed":"42","candidate_video":"/data/my_model/BallRoller/ctrl/42.mp4"}
```

**Still to implement and document as runnable commands:** full-suite
export/download/verify; dependency and capability preflight; generation-job
planning/execution; receipt validation; expected-job/metric completeness checks;
and the final reproducibility report. Do not present these as existing CLI
subcommands. The current `--dry-run` resolves discovered work; it does not prove
that all expected jobs exist or that CV/API evaluation will succeed.

The runner must then verify candidate integrity and condition identity, apply
the frozen frame alignment/normalization, load baked references, compute only
the required candidate-side features, run declared judges, and aggregate using
the locked metric policy. Check video decoding through the final expected frame,
not just file existence. Short/truncated clips must not be silently padded or
looped into successful long-horizon evaluations. Test retry/resume against a
changed candidate, changed reference, changed evaluator and interrupted output.

### E. What results users get

Retain the existing three outputs, extending their schemas explicitly rather
than breaking current consumers:

- `per_case_results.jsonl`: currently one row per model/case/seed/dimension,
  with status, metrics, `full_report`, timing and evaluator-lock summary. Add
  job/receipt, candidate/reference and environment hashes; normalization identity;
  per-metric value/unit/direction, source, status/reason, valid/expected counts;
  and evidence paths. Keep per-object/per-event/per-frame detail inspectable.
- `aggregate_summary.json`: retain per-case means, across-case statistics and
  within-case seed dispersion. Add expected/generated/evaluated/valid counts,
  missing and excluded identities, aggregation weights and repeat definitions.
  Use the same declared case set for model comparisons and show coverage beside
  scores. Never rank a partial supported subset as the full benchmark.
- `aggregate_summary.csv`: retain machine-readable dimension summaries, adding
  coverage/status columns and keeping metric units/direction discoverable.

New required companion outputs are `coverage.json` (every planned job and metric,
including absent candidates), `run_manifest.json` (complete invocation and
dependency identities), an `evidence/` tree (tracks/masks/selected frames and
auditable judge request/response artifacts with secrets redacted), and a concise
`report.md` linking results and failures. A fancy HTML dashboard is not P0.
Retain or link source candidates so a reported score can be traced to its video.

Define per-metric statuses for `ok`, `unsupported`, `missing_candidate`,
`invalid_candidate`, `not_applicable`, `unmeasurable`, and `error`, with reasons.
These are proposed output-schema additions, not current runner enum guarantees.
Refusals/rate limits are judge failures, not model scores. A genuine missing
object can still incur the metric's specified penalty; detection or calibration
failure must not automatically become that penalty. Nulls are never favorable
zeros. Keep causal LPIPS separate from temporal fallback and candidate seed
variance separate from judge variance. Publish no cross-dimension scalar unless
its normalization and weights have been explicitly approved.

### F. Concrete P0 hand-off and acceptance artifacts

Each P0 below must link its checked-in implementation, test command, and actual
acceptance output. These are required deliverables, not merely discussion topics.

| Item | Artifact/test that closes it |
| --- | --- |
| P0-01 | Signed-off roster/repeat policy, metric catalog, dataset/code licenses, asset-level redistribution inventory and dataset card. |
| P0-02 | Version-pinned public package + file manifest + exporter/downloader; anonymous download into a relocated clean directory verifies every file. |
| P0-03 | 19 reference manifests, explicit conditioning image for every case, frozen consumed frames and metric assets; provenance, reprojection/time/object-join checks and self-match tolerances pass. |
| P0-04 | 19 generation/action manifests including the four supplemental cases; schema validation and target/timing/capability rejection fixtures pass. |
| P0-05 | Adapter template + fixture + real example; complete expected-job plan, generation receipts, import/resume checks and runner-compatible candidates manifest. |
| P0-06 | Pinned full environment and CV asset lock; preflight passes with no private paths or warm developer caches and reports any API requirements. |
| P0-07 | Per-case normalization and metric-status schemas; tests for temporal offset, geometry mismatch, truncated video, missing anchors/labels and unsupported controls. |
| P0-08 | Frozen reference judgments/QA/rubrics and candidate judge audit fixture; repeat/refusal policy, redaction and explicit egress/cost opt-in. |
| P0-09 | Expected-versus-actual coverage gate, extended result artifacts and downloadable baseline candidates/scores; changed-byte resume and aggregation tests. |
| P0-10 | Published clean-machine transcript using the actual downloaded package: fixture generation, real candidate ingestion, all 8 dimensions, coverage checks, resume, and scores within declared tolerances. |

Additional omissions found while making the workflow concrete, all required
under the existing P0 IDs:

- [ ] **P0-03/04:** settle first-frame versus first-prediction indexing, action
  latency and reference/conditioning initial-state agreement for every case.
- [ ] **P0-04/05:** publish a capability/track matrix; prohibit silent camera
  versus object substitutions, dropped inputs, or text-only/I2V pooling.
- [ ] **P0-05:** test the generation input allowlist so future reference frames
  and evaluator answers do not leak through the adapter interface.
- [ ] **P0-02/03:** validate archive/path safety and safe array loading, and wire
  frozen reference frames into the actual consuming code path.
- [ ] **P0-07/09:** cover absent jobs as well as failed metric rows; preserve
  raw outputs, decoding failures, action compilation and normalization receipts.
- [ ] **P0-08/10:** document hosted-judge opt-in, credential redaction, repeat
  policy, and a clearly labeled no-API subset; do not promise offline all-metrics.

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
  *Progress 2026-10-08:* Table 4/7 trajectories have declared action targets
  and per-backend capability checks (`action_backends.yaml`). Remaining: carry
  target/frame semantics into a versioned per-case `actions.json`, migrate the
  19 Lite bundles (including the four supplemental cases), and replace the
  camera-only action on `RacingCar/prompt_third` for its kinematics assignment.

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
  *Progress 2026-10-08:* deterministic adapters, hashed receipts, queues and
  resume exist for Matrix-Game 2.0/3.0 and LingBot Base Cam on the Table 4/7
  track (`tools/model_adapters/`, `tools/prepare_*_inputs.py`,
  `tools/run_*`). These are model-specific; the model-neutral adapter
  template, no-cost fixture adapter and Lite job export are still missing.

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
  *Progress 2026-10-08:* `evaluator_assets.lock.json` pins four Table 4/7 CV
  checkpoints; extend it to RAFT, LPIPS/VGG, PTLFlow, OpenPIV and indoor depth
  and propagate it into the Lite case locks.

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
  Include the Unity-reference prompt-consistency scores and the causal-LPIPS
  reference events in the frozen fixtures (neither is in the bundles today).

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
  *Progress 2026-10-08:* one root `README.md` now carries installation,
  quickstart, suite matrix, action trajectories and every dimension's full
  evaluator reference. Make the test suite hermetic (no reads from ignored
  `runs/` or the sibling paper checkout) before wiring CI.

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
  Adopt one calibration policy for every backend (same probe set, acceptance
  bands, held-out non-scored validation scene); re-check the Matrix-Game 3.0
  gains off WanderingGarden; report realized yaw next to every consistency
  score; validate Matrix-Game 2.0 car/ball target control (thresholded WASD
  drops control magnitude); calibrate the Project Genie right turn.
- [ ] **Results owner (`multi-model`) — one repeat policy.** Reconcile LingBot's
  adaptive 5-to-30 expansion with the five-run protocol used elsewhere, keep
  the five-run cohort as the reported paper selection, and declare each
  model's native fps/duration and the applied resampling in every result row.
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
