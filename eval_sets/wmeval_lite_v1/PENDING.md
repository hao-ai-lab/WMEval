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

- **`ChessBoard/prompt_first`, `ChessBoard/prompt_third`** → `prompt_scene_consistency`
  Needs: `reference/ref.png`, `reference/unity_groundtruth.mp4`,
  `reference/artifacts/actions.txt` + `action.npy`/`intrinsics.npy`/`poses.npy`/`meta.json`,
  case.yaml `prompt:`/`rubric_scene_name:`. After bundling, run
  `generate-rubrics` (needs `OPENAI_API_KEY`) and commit the result.

- **`ChessBoard/lighting`** → `lighting_shading`
  Team decision first: if it judges the same video as `prompt_third`, just
  list `ChessBoard/prompt_third` under `lighting_shading.cases` (no new case
  needed). If it needs its own rollout, author one with
  `module_params.shadow_consistency.{shadow_text_prompt, shadow_anchor_label}`.

- **`SunRisesAndSets/lighting`** → `lighting_shading`
  Same bundle shape as a prompt case (prompt + ref.png + GT video + actions).
  Needs a protocol decision before authoring: the shadow-direction metrics
  (MOSD/SDJ) assume a static sun, but this scene's whole point is a moving
  sun — confirm how that should be scored first.

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

- **No real candidate world-model has been evaluated yet.** Every number in
  every live case bundle is reference-vs-reference self-consistency —
  it validates the pipeline, not an actual model's output.
- **`kinematics_mechanics`'s mechanical quantities** (momentum/kinetic-energy
  exact-match) inherit the monocular depth model's domain-gap bias and are
  only meaningfully accurate for objects with real, non-trivial motion.
- **`MirrorRollingBall`'s depth accuracy** relies on an apparent-size
  (photogrammetric) workaround, not a monocular depth model — no monocular
  depth model has been found that handles its reflective sphere correctly.
