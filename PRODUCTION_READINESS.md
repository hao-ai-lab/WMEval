# WMEval production-readiness audit

Generated from the live suite, case bundles, trajectory inventories, and `experiments/evaluation_dimension_standardization.yaml`.

A valid case bundle is not automatically a portable or paper-parity experiment. The table below keeps those concepts separate.

## Snapshot

- 33 validated case bundles across 8 dimensions.
- 32 cases have native machine-readable trajectories; 1 rely on supplemental reconstructions.
- Project Genie backend readiness: `{"native_executable_ready": 32, "supplemental_executable_ready": 1}`.
- Open gaps by severity: `{"critical": 0, "high": 8, "low": 3, "medium": 5}`.
- Open gaps by release priority: `{"P0": 4, "P1": 7, "P2": 5}`.

## Dimension comparison

| Dimension | Readiness | Cases | Native / supplemental trajectories | Genie-executable | Normalization | P0 / P1 / P2 |
|---|---:|---:|---:|---:|---|---:|
| `lpips` | partial | 7 | 7 / 0 | 7/7 | `beamdomino_native`, `liquid_832x464_16fps`, `none` | 0 / 0 / 0 |
| `action_controllability` | partial | 3 | 3 / 0 | 3/3 | `beamdomino_native`, `none` | 0 / 1 / 0 |
| `prompt_scene_consistency` | partial | 6 | 6 / 0 | 6/6 | `none` | 0 / 2 / 0 |
| `lighting_shading` | ready | 1 | 1 / 0 | 1/1 | `none` | 0 / 0 / 2 |
| `environment_object_consistency` | partial | 16 | 16 / 0 | 16/16 | `match_reference`, `none` | 1 / 1 / 0 |
| `kinematics_mechanics` | partial | 3 | 3 / 0 | 3/3 | `match_reference`, `none` | 0 / 2 / 0 |
| `physical_causality` | partial | 3 | 3 / 0 | 3/3 | `beamdomino_native` | 0 / 1 / 0 |
| `fluid_mechanics` | partial | 4 | 4 / 0 | 4/4 | `liquid_832x464_16fps` | 0 / 0 / 1 |

## Cross-cutting backlog

- **P0 / HIGH — `action_schema`:** Native action.npy arrays, JSON browser operations, structured prose, and supplemental reconstructions coexist without one versioned action schema or explicit channel semantics.
  Remediation: Resolved: all 32 live cases bundle a versioned trajectory.yaml (time base, channels, units) with compiled action.npz / actions.jsonl, and every trajectory id declares its target entity in action_backends.yaml; supplemental controls were migrated into the bundles.
- **P0 / HIGH — `candidate_normalization`:** Most non-liquid cases use candidate_normalization none even when metrics compare candidate frame f with reference frame f.
  Remediation: Every row records input_contract (candidate vs. reference fps, size, frames, duration) and flags unnormalized rows outside the reference spec. Normalization is not added where it would change paper-parity scores; decide per case before adding it.
- **P0 / HIGH — `dependency_lock_content`:** Live case evaluator locks have dependency_versions but empty artifact_schema_versions, and they describe the macOS environment only.
  Remediation: dependency_versions are filled for every live case (tools/lock_case_dependencies.py, macOS venv). Remaining: populate artifact_schema_versions and add a CUDA-environment lock set.
- **P0 / HIGH — `external_model_artifacts`:** Candidate-side VLM judging depends on a hosted provider snapshot; model weights are pinned.
  Remediation: Checkpoints are pinned with revisions and sha256 (evaluator_assets.lock.json) and checked by tools/preflight_lite.py; reference-side judgments are frozen fixtures. Remaining: candidate-side hosted judging depends on the provider's model snapshot.
- **P0 / MEDIUM — `result_status_schema`:** Metric availability, primary source, tracked-label coverage, and diagnostic-only status are not represented uniformly across dimensions.
  Remediation: Resolved: rows carry metric_status (ok / unmeasurable / row status), action_support, input_contract and lpips_primary_source; aggregates report metric_status_counts, lpips_temporal_fallback_rate and num_unsupported.
- **P1 / MEDIUM — `ci_release_gate`:** Bundle validation and tests are local commands; no checked-in CI/release gate verifies the suite, readiness manifest, lightweight tests, and artifact hashes together.
  Remediation: Resolved: .github/workflows/ci.yml validates the suite and runs the hermetic tests (integration tests skip without local data).
- **P1 / MEDIUM — `repeat_semantics`:** Five independent model generations, repeated VLM judge passes, and cached reference judgments are different sampling units but are not described by one aggregation schema.
  Remediation: Resolved: suite.yaml repeat_policy (required_repeats 5) gates the headline; judge passes and frozen reference judgments are documented as separate sampling units in DIFFERENCES_FROM_PAPER.md and FROZEN_JUDGE_OUTPUTS.md.
- **P2 / HIGH — `install_contract`:** The tested macOS environment is pinned in requirements/macos.lock, while there is no pyproject.toml/uv.lock project contract and CUDA dependencies are embedded separately in a Modal script.
  Remediation: Define supported CPU/macOS and CUDA extras in project metadata, produce uv locks for each platform, and add an offline checkpoint preflight.
- **P2 / LOW — `artifact_retention`:** Local runs dominate repository disk usage, but there is no documented retention/archive policy separating canonical evidence from resumable caches and disposable diagnostics.
  Remediation: Define canonical run manifests and checksums, then provide a dry-run cleanup command for derived caches and superseded reports.

## Dimension-specific backlog

### lpips


### action_controllability

- **P1 / MEDIUM:** The pixel-EPE headline is resolution dependent despite relative EPE being available.
  Remediation: Freeze resolution normalization or promote a scale-normalized metric with an explicit direction.

### prompt_scene_consistency

- **P1 / HIGH:** Exact unpublished generation prompts are unavailable for paper-parity reruns.
  Remediation: Mark prompts as reconstructed in every run receipt and retain exact prompt hashes for all new runs.
- **P1 / MEDIUM:** Hosted VLM output depends on a mutable provider/model alias even with a pinned rubric version.
  Remediation: Record API/model snapshot metadata and maintain a frozen regression fixture with expected tolerance bands.

### lighting_shading

- **P2 / MEDIUM:** The fixed-sun paper protocol and optional moving-sun interpretation require different metrics, but only the fixed-sun protocol is implemented.
  Remediation: Keep the protocols separate and add sweep-aware scoring only as a new case.
- **P2 / LOW:** ChessBoard lighting coverage is optional and not bundled.
  Remediation: Decide whether cross-scene lighting is in release scope before adding a locked case.

### environment_object_consistency

- **P0 / HIGH:** Candidate fps, resolution, duration, and frame alignment are not normalized by the suite contract.
  Remediation: Table 7 cells use match_reference; the remaining cases record input_contract and flag rows outside the reference spec (within_spec false). Normalizing them would change scores; decide per case.
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

### fluid_mechanics

- **P2 / LOW:** The dimension has no scientifically valid single-scalar summary, so a headline can obscure per-axis reversals.
  Remediation: Treat the per-profile multi-axis panel as canonical and label the scalar headline as navigation only.

## Reproduce

```bash
uv run python tools/run_wmeval_lite.py validate --suite eval_sets/wmeval_lite_v1
uv run python tools/audit_production_readiness.py --markdown PRODUCTION_READINESS.md --json reports/production_readiness.json
```

`--fail-on-high` is intended as a future release gate; it currently fails because the documented high-severity backlog is still open.
