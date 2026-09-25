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

- **`BeamDomino/causal`** → `physical_causality`
  Reference video + actions are already bundled. Missing only
  `reference/artifacts/qa_dataset.json` (CausalStep stepwise QA). Then wire
  `physical_causality` into `suite.yaml` (registry, modules `[causal_step]`,
  headline from the csr/delta family). `causal_glancing` and
  `causal_direct_strike` variants each need their own video + actions + QA.

- **`LiquidContainer`** (`fluid_rest`, `fluid_step_hold`, `fluid_impulse_decay`,
  `fluid_chirp_strafe`) → `fluid_mechanics`
  Needs `reference/reference.mp4` + `liquid_mask.npz`/`bottle_mask.npz`/
  `raft_pooled_flow.npz`. The `.npz` artifacts **must** be produced via
  `evaluation/lite/reference_artifacts.write_*` (PR #19's API) or the
  checksum lock rejects them — coordinate with @abhilashreddys before
  authoring. case.yaml needs `protocol.candidate_normalization:
  liquid_832x464_16fps`; `suite.yaml` gains the `fluid_mechanics` dimension
  (4 liquid modules).

## Known limitations on live dimensions

- **Real-candidate coverage.** `lighting_shading` was run on the paper's
  6 models x 5 seeds (2026-09-15, reproduces the paper exactly), and
  Waypoint-1.5 has been run through every live dimension: the non-VLM ones
  on 2026-09-22 (55 rows, no errors) and `prompt_scene_consistency` +
  `lpips` on 2026-09-24 (30 + 10 rows, all completed, ~3M gpt-5.4 input
  tokens in total). Numbers are in the suite.yaml banners. Waypoint is a
  first-person, keyboard/mouse-only model, so third-person and object-control
  cases measure its action-target ambiguity rather than a strong result.
- **`lpips` mixes causal and temporal numbers without a row-level flag.**
  When the judge matches no contact events (every BeamDomino/causal row for
  Waypoint: 0 candidate contacts), the row still completes with
  `lpips_primary_source: temporal_only` inside `full_report` and a headline
  `lpips_mean` that is the temporal baseline, so the aggregate averages
  causal and temporal LPIPS as if they were one metric. Surface the source
  (or count unmatched rows as a separate status) in `extract_metrics`.
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
- **`kinematics_mechanics` drops untracked labels silently.** The headline
  `fde_px_mean`/`dtw_px_mean` average over whichever labels ByteTrack kept:
  BilliardGame/prompt_first seeds 2/4 report 0.5 px because only the static
  "red ball" survived, while the seeds that also tracked the rolling
  "yellow ball" report ~365 px for it. An untracked label should count as a
  failure (or the row should carry an explicit `labels_tracked`), not
  vanish from the mean.
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
