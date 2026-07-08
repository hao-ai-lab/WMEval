# Pending case data — intake checklist

Deliverables for each planned-but-not-yet-bundled case. Case folders are
created **only when the data lands** (the trust layer rejects placeholders:
everything under `cases/` must validate). Wiring steps for every case:

1. `mkdir -p cases/<Scene>/<variant>/reference/artifacts` and drop the files
2. write `case.yaml` (copy a live case of the same kind as template)
3. `python tools/build_lite_case.py cases/<Scene>/<variant>` (checksums + lock)
4. add the case id to the dimension's `cases:` list in `suite.yaml`
5. `python tools/run_wmeval_lite.py validate` (+ `generate-rubrics` where noted)

Delete each section (and finally this file) as data lands.

---

## ChessBoard — prompt_scene_consistency + lighting_shading

**`ChessBoard/prompt_first`, `ChessBoard/prompt_third`**

| file | notes |
| --- | --- |
| `reference/ref.png` | first frame |
| `reference/unity_groundtruth.mp4` | GT rollout (smoke-test candidate) |
| `reference/artifacts/actions_first.txt` / `actions_third.txt` | keyboard script that drove the rollout (camera moves through the scene) |
| `reference/artifacts/action.npy` (+ `intrinsics.npy`, `poses.npy`, `meta.json`) | numeric action data |
| case.yaml `prompt:` | scene description text |
| case.yaml `rubric_scene_name:` | `ChessBoard_first` / `ChessBoard_third` |

After bundling: `run_wmeval_lite.py generate-rubrics` (needs `OPENAI_API_KEY`), commit the rubrics.

**`ChessBoard/lighting`** — team decision first: if lighting judges the same
video as `prompt_third`, do NOT create a case — just list
`ChessBoard/prompt_third` under `lighting_shading.cases` (borrow). If lighting
needs its own rollout, create the case with
`module_params.shadow_consistency.{shadow_text_prompt, shadow_anchor_label}`
(e.g. chess pieces as shadow casters).

## SunRisesAndSets — lighting_shading

**`SunRisesAndSets/lighting`** (third person)

Same bundle shape as a prompt case (prompt + ref.png + GT video + actions) plus
`module_params.shadow_consistency` detection prompts. Note: the sun moving is
the point of this scene — shadow-direction metrics (MOSD/SDJ) assume a static
sun; confirm with the dimension owner how the moving-sun protocol is scored
before authoring.

## WanderingGarden — scene_character_consistency

**`WanderingGarden/scene_first`, `WanderingGarden/scene_third`**

Bundle shape as prompt cases. Blocked on defining the dimension itself:
which judge configuration (the PR #14 vlm_consistency pipeline is the
candidate), what the rubrics are generated from, and its suite.yaml
`vlm:`/aggregate spec.

## BeamDomino — physical_causality

**`BeamDomino/causal`** — reference video + actions are ALREADY bundled
(earmarked). Missing only:

| file | notes |
| --- | --- |
| `reference/artifacts/qa_dataset.json` | CausalStep stepwise QA over the rollout |

Then: re-run `build_lite_case.py`, add the `physical_causality` dimension to
suite.yaml (registry, modules `[causal_step]`, ref-based, headline from the
csr/delta family). More strike variants (`causal_glancing`,
`causal_direct_strike`) each need their own reference.mp4 + actions + QA.

## LiquidContainer — fluid_mechanics

**`LiquidContainer/fluid_rest`, `fluid_step_hold`, `fluid_impulse_decay`, `fluid_chirp_strafe`**

| file | notes |
| --- | --- |
| `reference/reference.mp4` | sealed-bottle rollout, protocol-locked geometry |
| `reference/artifacts/liquid_mask.npz` | SAM2 per-frame liquid mask |
| `reference/artifacts/bottle_mask.npz` | SAM2 bottle mask |
| `reference/artifacts/raft_pooled_flow.npz` | pooled RAFT flow field |

The `.npz` artifacts MUST be produced with
`evaluation/lite/reference_artifacts.write_*` (PR #19's API) or the
`evaluator_lock` consistency check refuses them — coordinate with
@abhilashreddys. case.yaml `protocol.candidate_normalization:
liquid_832x464_16fps`. suite.yaml gains the `fluid_mechanics` dimension
(4 liquid modules).

## BallRoller / RacingCar — kinematics_mechanics

Blocked on module confirmation (`motion` proposed). Then either own rollouts
(`BallRoller/kinematics`, `RacingCar/kinematics`) or borrow
`BallRoller/ctrl` + `RacingCar/prompt_third` (declare its bundled
`unity_groundtruth.mp4` as `reference.video`). Plus a `motion` metric
extractor in `evaluation/lite/aggregate.py`.

## BilliardGame

In the scene roster but unmapped to any dimension — confirm intent before
authoring anything.
