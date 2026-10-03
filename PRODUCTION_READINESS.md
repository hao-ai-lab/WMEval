# WMEval production-readiness audit

Generated from the live suite, case bundles, trajectory inventories, and `experiments/evaluation_dimension_standardization.yaml`.

A valid case bundle is not automatically a portable or paper-parity experiment. The table below keeps those concepts separate.

## Snapshot

- 19 validated case bundles across 8 dimensions.
- 15 cases have native machine-readable trajectories; 4 rely on supplemental reconstructions.
- Project Genie backend readiness: `{"native_executable_ready": 9, "not_machine_executable": 7, "supplemental_executable_ready": 3}`.
- Open gaps by severity: `{"critical": 0, "high": 16, "low": 3, "medium": 11}`.
- Open gaps by release priority: `{"P0": 11, "P1": 14, "P2": 5}`.

## Dimension comparison

| Dimension | Readiness | Cases | Native / supplemental trajectories | Genie-executable | Normalization | P0 / P1 / P2 |
|---|---:|---:|---:|---:|---|---:|
| `lpips` | partial | 6 | 6 / 0 | 1/6 | `beamdomino_native`, `liquid_832x464_16fps`, `none` | 1 / 1 / 0 |
| `action_controllability` | partial | 2 | 2 / 0 | 1/2 | `beamdomino_native`, `none` | 1 / 2 / 0 |
| `prompt_scene_consistency` | partial | 6 | 6 / 0 | 6/6 | `none` | 0 / 2 / 0 |
| `lighting_shading` | partial | 1 | 0 / 1 | 1/1 | `none` | 0 / 1 / 2 |
| `environment_object_consistency` | partial | 6 | 3 / 3 | 6/6 | `none` | 3 / 1 / 0 |
| `kinematics_mechanics` | partial | 3 | 2 / 1 | 3/3 | `none` | 0 / 2 / 0 |
| `physical_causality` | partial | 3 | 3 / 0 | 0/3 | `beamdomino_native` | 0 / 2 / 0 |
| `fluid_mechanics` | partial | 4 | 4 / 0 | 0/4 | `liquid_832x464_16fps` | 1 / 1 / 1 |

## Cross-cutting backlog

- **P0 / HIGH — `action_schema`:** Native action.npy arrays, JSON browser operations, structured prose, and supplemental reconstructions coexist without one versioned action schema or explicit channel semantics.
  Remediation: Introduce a versioned trajectory manifest with time base, channel names, units, target entity, view, and backend adapter status; migrate supplemental controls into locked bundles.
- **P0 / HIGH — `candidate_normalization`:** Most non-liquid cases use candidate_normalization none even when metrics compare candidate frame f with reference frame f.
  Remediation: Standardize fps, duration, resolution, crop, and time-base normalization in the runner and record the transform in every result row.
- **P0 / HIGH — `dependency_lock_content`:** Case evaluator locks contain evaluator fingerprints, but dependency_versions and artifact_schema_versions are empty in every live case.
  Remediation: Populate machine-verifiable dependency and artifact-schema versions, then refresh evaluator fingerprints.
- **P0 / HIGH — `external_model_artifacts`:** Several evaluators resolve model aliases or download checkpoints at runtime, and VLM-backed dimensions depend on a mutable hosted API.
  Remediation: Pin checkpoint revisions and hashes, cache immutable artifacts, record provider/model snapshots, and retain deterministic fixtures for reference-side judgments.
- **P0 / MEDIUM — `result_status_schema`:** Metric availability, primary source, tracked-label coverage, and diagnostic-only status are not represented uniformly across dimensions.
  Remediation: Add a shared result-envelope schema with status, reason, source, units, coverage, evaluator fingerprint, and candidate-normalization provenance per metric.
- **P1 / MEDIUM — `ci_release_gate`:** Bundle validation and tests are local commands; no checked-in CI/release gate verifies the suite, readiness manifest, lightweight tests, and artifact hashes together.
  Remediation: Add a lightweight CI job plus an opt-in GPU/API integration workflow and publish the readiness JSON with releases.
- **P1 / MEDIUM — `repeat_semantics`:** Five independent model generations, repeated VLM judge passes, and cached reference judgments are different sampling units but are not described by one aggregation schema.
  Remediation: Record generation_seed, generation_repeat, judge_repeat, aggregation_axis, ddof, and effective sample size separately.
- **P2 / HIGH — `install_contract`:** The tested macOS environment is pinned in requirements-macos.lock, while there is no pyproject.toml/uv.lock project contract and CUDA dependencies are embedded separately in a Modal script.
  Remediation: Define supported CPU/macOS and CUDA extras in project metadata, produce uv locks for each platform, and add an offline checkpoint preflight.
- **P2 / LOW — `artifact_retention`:** Local runs dominate repository disk usage, but there is no documented retention/archive policy separating canonical evidence from resumable caches and disposable diagnostics.
  Remediation: Define canonical run manifests and checksums, then provide a dry-run cleanup command for derived caches and superseded reports.

## Dimension-specific backlog

### lpips

- **P0 / HIGH:** Reference-side causal event judgment is not a single cached fixture for every case and may vary across candidate rows.
  Remediation: Cache reference events per case, judge model, prompt version, and frame-sampling contract.
- **P1 / MEDIUM:** BeamDomino and liquid action descriptions are numeric/structured but not native browser operations.
  Remediation: Add explicit target-aware browser adapters or declare those model backends unsupported in the case contract.

### action_controllability

- **P0 / HIGH:** The control target is ambiguous for keyboard-only models, so ego-camera motion can be scored against an intended object action.
  Remediation: Bind every action channel to a target entity and report unsupported target channels instead of a numeric failure score.
- **P1 / HIGH:** Synthetic-flow calibration depends on an external PTLFlow checkout resolved outside this repository.
  Remediation: Vendor or package the required synthetic-flow interface and checksum its calibration artifact.
- **P1 / MEDIUM:** The pixel-EPE headline is resolution dependent despite relative EPE being available.
  Remediation: Freeze resolution normalization or promote a scale-normalized metric with an explicit direction.

### prompt_scene_consistency

- **P1 / HIGH:** Exact unpublished generation prompts are unavailable for paper-parity reruns.
  Remediation: Mark prompts as reconstructed in every run receipt and retain exact prompt hashes for all new runs.
- **P1 / MEDIUM:** Hosted VLM output depends on a mutable provider/model alias even with a pinned rubric version.
  Remediation: Record API/model snapshot metadata and maintain a frozen regression fixture with expected tolerance bands.

### lighting_shading

- **P1 / HIGH:** The orbit has an exact browser schedule but only a semantic reconstructed numeric trajectory, so Unity replay is not exact.
  Remediation: Capture and lock the original Unity controller trace with camera poses and timestamps.
- **P2 / MEDIUM:** The fixed-sun paper protocol and optional moving-sun interpretation require different metrics, but only the fixed-sun protocol is implemented.
  Remediation: Keep the protocols separate and add sweep-aware scoring only as a new case.
- **P2 / LOW:** ChessBoard lighting coverage is optional and not bundled.
  Remediation: Decide whether cross-scene lighting is in release scope before adding a locked case.

### environment_object_consistency

- **P0 / HIGH:** BilliardGame, ChessBoard walk, and WanderingGarden rely on supplemental trajectories rather than native locked controls.
  Remediation: Move verified controls into each case bundle and refresh checksums/evaluator locks.
- **P0 / HIGH:** Candidate fps, resolution, duration, and frame alignment are not normalized by the suite contract.
  Remediation: Add suite-level video normalization and record the applied transform.
- **P0 / HIGH:** Pairwise-distance MAE needs at least two stable, visible anchors, but anchor visibility/coverage is not a uniform validity gate.
  Remediation: Require per-anchor visibility, assignment confidence, and minimum paired-frame coverage before reporting the metric.
- **P1 / MEDIUM:** Reflective MirrorRollingBall depth uses a photogrammetric workaround and other cases depend on scene-specific monocular-depth model choices.
  Remediation: Record depth source per metric and maintain per-scene self-match calibration bounds.

### kinematics_mechanics

- **P1 / HIGH:** Paper-native metric units are available only for calibrated planar captures; legacy cases retain pixel/depth diagnostics.
  Remediation: Add per-scene world-coordinate calibration with held-out reprojection validation and explicit unit labels.
- **P1 / MEDIUM:** Relative momentum/energy error is unstable near zero ground-truth motion, and exact-match rounding is highly threshold sensitive.
  Remediation: Keep error rate as the headline, add an absolute-error floor policy, and treat significant-digit exact match as diagnostic.

### physical_causality

- **P1 / HIGH:** Candidate-side VLM judgments remain stochastic and depend on a hosted model alias; temperature zero is not deterministic.
  Remediation: Record all judge passes, model snapshot metadata, and uncertainty; re-baseline whenever the judge changes.
- **P1 / MEDIUM:** BeamDomino trajectories are numeric with structured text but have no target-aware Project Genie browser program.
  Remediation: Implement the object-strike control adapter or explicitly mark browser generation unsupported.

### fluid_mechanics

- **P0 / MEDIUM:** Candidate-side SAM2/RAFT checkpoints are external runtime artifacts whose revisions/hashes are not captured in case locks.
  Remediation: Pin and checksum checkpoint revisions and include them in result provenance.
- **P1 / HIGH:** Numeric Unity action profiles lack browser-executable, target-aware controls for hosted world models.
  Remediation: Define a bottle/camera action adapter with units and reject models that cannot express it.
- **P2 / LOW:** The dimension has no scientifically valid single-scalar summary, so a headline can obscure per-axis reversals.
  Remediation: Treat the per-profile multi-axis panel as canonical and label the scalar headline as navigation only.

## Reproduce

```bash
uv run python tools/run_wmeval_lite.py validate --suite eval_sets/wmeval_lite_v1
uv run python tools/audit_production_readiness.py --markdown PRODUCTION_READINESS.md --json reports/production_readiness.json
```

`--fail-on-high` is intended as a future release gate; it currently fails because the documented high-severity backlog is still open.
