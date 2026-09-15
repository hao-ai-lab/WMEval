# Evaluation Sets

Curated, Unity-free benchmark bundles. Reference Unity videos and artifacts ship
with the repo so evaluators never need a local Unity checkout.

## WMEval Lite v1

Path: `eval_sets/wmeval_lite_v1/`

One suite, one CLI, every dimension. `suite.yaml` declares the dimensions;
each case ships as a checksummed bundle under `cases/<scene>/<variant>/`.

### Naming convention

- **scene** — the Unity scene's CamelCase name, from the fixed roster:
  `BallRoller`, `BeamDomino`, `BilliardGame`, `ChessBoard`,
  `MirrorRollingBall`, `LiquidContainer`, `RacingCar`, `SunRisesAndSets`,
  `WanderingGarden`.
- **variant** — `<dimension-tag>[_<perspective>][_<setting>]`:
  - dimension tags: `lpips`, `ctrl` (action_controllability), `prompt`
    (prompt_scene_consistency), `lighting`, `kinematics`, `scene`
    (scene_character_consistency), `causal` (physical_causality), `fluid`
  - perspective (`first`/`third`) appears only for the two-view dimensions
    (`prompt`, `scene`); every other dimension is third-person by rule
  - setting suffix when the protocol varies: `collision`, `canonical`,
    `glancing`, `rest`, `step_hold`, ...
  - examples: `MirrorRollingBall/lpips_collision`, `RacingCar/prompt_first`,
    `BallRoller/ctrl`, `LiquidContainer/fluid_rest`

A **case is one atomic setup** (one prompt + one reference rollout + one
action protocol) owned by the dimension its variant names. When another
dimension evaluates the same video, suite.yaml lists the case under both
dimensions (**borrowing** — the variant keeps the owner's name and the
borrow is visible in the case × dimension matrix). Example:
`RacingCar/prompt_third` is owned by `prompt_scene_consistency` and borrowed
by `environment_object_consistency` and `kinematics_mechanics`.

**Generation conditions are uniform across all cases**: every case bundles a
`prompt` (case.yaml), a first frame (`reference/ref.png` or frame 0 of
`reference/reference.mp4`), and an **action script**
(`reference/artifacts/actions_*.txt`, numeric form `action.npy`). Candidates
must be generated following the case's action script — including the
prompt/shadow-consistency cases, whose actions move the camera through the
scene so consistency is judged across views, not on a static shot. Dimensions
differ only in what the **evaluator** consumes:

Dimension ids follow the paper taxonomy. The authoritative target
scene × dimension mapping:

| Scene | Dimensions (target) | Status |
| --- | --- | --- |
| MirrorRollingBall | lpips, prompt_scene_consistency | ✅ live |
| BallRoller | action_controllability, kinematics_mechanics, environment_object_consistency | ctrl + environment_object_consistency (object-only) live; kinematics blocked (data-provenance conflict, see PENDING.md) |
| RacingCar | prompt_scene_consistency, kinematics_mechanics | prompt live; kinematics pending |
| ChessBoard | prompt_scene_consistency, lighting_shading | prompt data pending; lighting pending a team decision (see PENDING.md); `ChessBoard/walk_third` live under environment_object_consistency |
| SunRisesAndSets | lighting_shading | ✅ live (`SunRisesAndSets/lighting`, first-person fixed-sun walking orbit — the paper regime) |
| WanderingGarden | scene_character_consistency | data pending |
| LiquidContainer | fluid_mechanics | data pending (PR #19 artifacts) |
| BeamDomino | physical_causality | data pending (qa_dataset); `cases/BeamDomino/causal` earmarked |
| BilliardGame | (unmapped — confirm with team) | — |

Currently wired dimensions:

| Dimension | Evaluator | Evaluation-side reference | Cases |
| --- | --- | --- | --- |
| `lpips` | registry (`lpips`, causal VLM collision) | reference video | MirrorRollingBall/lpips_collision |
| `action_controllability` | registry (`control_motion`) | reference video + `action.npy` | BallRoller/ctrl |
| `prompt_scene_consistency` | vlm_consistency judge | none (prompt + first frame only) | MirrorRollingBall + RacingCar, prompt_first/prompt_third |
| `lighting_shading` | registry (`shadow_consistency`) | none (candidate-internal physics; the bundle ships the analytic fixed-sun orbit log as the geometry floor) | SunRisesAndSets/lighting (ChessBoard still pending) |
| `environment_object_consistency` | registry (`environment_object_consistency`) | reference video + Unity object transforms | BallRoller/ctrl (object-only: static ramp/dome; character half not attempted, see PENDING.md) |

### Case × dimension matrix (current)

| case | lpips | action_controllability | prompt_scene_consistency | lighting_shading | environment_object_consistency | kinematics_mechanics |
| --- | :-: | :-: | :-: | :-: | :-: | :-: |
| BallRoller/ctrl | | ✓ | | | ✓ | ✓ |
| BilliardGame/prompt_first | | | | | ✓ | ✓ |
| ChessBoard/walk_third | | | | | ✓ | |
| MirrorRollingBall/lpips_collision | ✓ | | | | | |
| MirrorRollingBall/prompt_first | | | ✓ | | | |
| MirrorRollingBall/prompt_third | | | ✓ | | ✓ | |
| RacingCar/prompt_first | | | ✓ | | | |
| RacingCar/prompt_third | | | ✓ | | ✓ | ✓ |
| SunRisesAndSets/lighting | | | | ✓ | | |
| WanderingGarden/scene_first | | | | | ✓ | |

(`suite.yaml` is the single source of truth; `list-cases` prints this matrix
from it. `cases/BeamDomino/causal` is bundled but currently unreferenced —
earmarked for `physical_causality`. Case folders are created only when their
data lands; per-case deliverables for everything pending are listed in
[`wmeval_lite_v1/PENDING.md`](wmeval_lite_v1/PENDING.md).)

### Layout

```
eval_sets/wmeval_lite_v1/
  suite.yaml                       # dimension-level manifest (single source of truth)
  cases/<scene>/<variant>/
    case.yaml                      # schema_version 2: prompt, modules, protocol, evaluator_lock
    reference/
      reference.mp4                # reference video (reference-based dimensions)
      ref.png                      # first frame (reference-free prompt_scene_consistency)
      checksums.sha256             # covers every file under reference/ (tamper check;
                                   #   feeds the evaluator fingerprint)
      baseline.json                # OPTIONAL: real reference baseline scores only —
                                   #   no empty placeholders
      artifacts/
        actions.txt                # keyboard action script (standard name)
        action.npy                 # numeric actions (+intrinsics.npy/poses.npy)
        rubrics/<SceneName>/<provider>_<model>_<version>.json
                                   # committed judge rubrics (vlm dimensions) —
                                   #   inside the bundle, so checksummed
        qa_dataset.json / *.npz    # dimension-specific payloads (causal/fluid)
```

### Quick start

```bash
# What does the suite contain?
python tools/run_wmeval_lite.py list-cases --suite eval_sets/wmeval_lite_v1

# Validate every case bundle (SHA256 + completeness)
python tools/run_wmeval_lite.py validate --suite eval_sets/wmeval_lite_v1

# Evaluate a model: drop rollouts at <model_root>/<scene>/<variant>/<seed>.mp4
python tools/run_wmeval_lite.py run \
  --suite eval_sets/wmeval_lite_v1 \
  --model-root runs/genie3 \
  --output-dir reports/wmeval_lite/genie3

# On Modal GPUs, one container per candidate (defaults to the paper's
# lighting rollouts on the WMEval-lighting-repro volume; see the docstring)
modal run tools/run_wmeval_lite_modal.py --dimensions lighting_shading

# Subset of dimensions, resumable
python tools/run_wmeval_lite.py run \
  --suite eval_sets/wmeval_lite_v1 \
  --model-root runs/genie3 \
  --dimensions lpips,action_controllability \
  --resume \
  --output-dir reports/wmeval_lite/genie3
```

Candidate input is either the directory layout above (model name = directory
name; optional `model.yaml` for per-model params such as `shadow_hud_preset`)
or a strict JSONL manifest:

```json
{"model":"genie3","case_id":"MirrorRollingBall/lpips_collision","seed":"0001","candidate_video":"/path/to/rollout.mp4"}
{"model":"genie3","case_id":"RacingCar/prompt_first","seed":"0001","candidate_video":"/path/to/other.mp4","params":{"shadow_hud_preset":"genie"}}
```

Legacy ids from the pre-suite manifests (e.g. `sample_scene`,
`mirror_ball_third_person`, and the retired snake_case case ids) are recorded
in `suite.yaml` under `legacy_case_aliases`; convert old candidate files with
`tools/convert_legacy_candidates.py`.

### Output

`--output-dir` receives:

- `per_case_results.jsonl` — one row per (dimension, case, model, seed):
  `status`, flat `metrics`, the untouched module `full_report`, and the case's
  `evaluator_lock` stamp.
- `aggregate_summary.json` — per model × dimension: two-level means
  (seeds within a case, then across cases) **with dispersion** — per metric
  `stats` carry `std_across_cases`, `min/max_case_mean`, the raw `case_means`,
  and per-case seed stds (`seed_std_by_case`, null for single-seed cases).
  No case-level bootstrap/CI is fabricated at these case counts; report
  mean ± std_across_cases and the case means themselves.
- `aggregate_summary.csv` — rows = models; per dimension two columns:
  headline mean and `_std` (std across cases; empty when only one case).

### Authoring cases

Reference files in place → `python tools/build_lite_case.py <case_dir>` writes
`checksums.sha256` and fills `evaluator_lock` (git revision + fingerprint).
Prompt-consistency rubrics: `python tools/run_wmeval_lite.py generate-rubrics`
(requires `OPENAI_API_KEY`) writes each rubric into its case bundle and
re-authors the case (checksums + lock) automatically. Rubrics are committed:
generation is an authoring step, not part of evaluation — and since the
rubric is checksummed, changing the judging basis visibly changes the case
fingerprint.
