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

- **Only `lighting_shading` has been run on real world-model candidates.**
  `SunRisesAndSets/lighting` was evaluated on the paper's 6 models x 5 seeds
  (2026-09-15) and reproduces the paper's per-run numbers exactly (see the
  suite.yaml banner). Every number quoted for the other live dimensions is
  still reference-vs-reference self-consistency — it validates the pipeline,
  not an actual model's output.
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
