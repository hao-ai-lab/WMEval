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
- **variant** — lowercase snake describing the evaluation setup: perspective
  (`first`, `third`) plus a rollout/protocol suffix when one scene carries
  several setups (`third_collision`, and later `canonical`, `rest`,
  `step_hold`, ...).

A **case is one atomic setup** (one prompt + one reference rollout + one
action protocol), not a scene: dimensions share a case only when a single
candidate video can serve them all. When dimensions need different rollouts
of the same scene, they are different variants — e.g. `MirrorRollingBall`
currently has `first`/`third` (look-around rollouts, prompt/shadow
consistency) and `third_collision` (collision rollout, causal LPIPS +
controllability).

**Generation conditions are uniform across all cases**: every case bundles a
`prompt` (case.yaml), a first frame (`reference/ref.png` or frame 0 of
`reference/reference.mp4`), and an **action script**
(`reference/artifacts/actions_*.txt`, numeric form `action.npy`). Candidates
must be generated following the case's action script — including the
prompt/shadow-consistency cases, whose actions move the camera through the
scene so consistency is judged across views, not on a static shot. Dimensions
differ only in what the **evaluator** consumes:

| Dimension | Evaluator | Evaluation-side reference | Cases |
| --- | --- | --- | --- |
| `causal_lpips` | registry (`lpips`, causal VLM collision) | reference video | MirrorRollingBall/third_collision, BallRoller/third, BeamDomino/third |
| `controllability` | registry (`control_motion`) | reference video + `action.npy` | same three as causal_lpips |
| `prompt_consistency` | vlm_consistency judge | none (prompt + first frame only) | MirrorRollingBall + RacingCar, first/third |
| `shadow_consistency` | registry (`shadow_consistency`) | none (candidate-internal physics) | MirrorRollingBall first/third, RacingCar/third |
| `causal_step` | registry (`causal_step`) | reference video + QA dataset | BeamDomino canonical/glancing/direct_strike (data pending) |
| `liquid` | registry (4 liquid modules) | reference video + `.npz` artifacts | LiquidContainer rest/step_hold/impulse_decay/chirp_strafe (data pending) |

(`suite.yaml` is the single source of truth; run `list-cases` for the current
state.)

### Layout

```
eval_sets/wmeval_lite_v1/
  suite.yaml                       # dimension-level manifest (single source of truth)
  cases/<scene>/<variant>/
    case.yaml                      # schema_version 2: prompt, modules, protocol, evaluator_lock
    reference/
      reference.mp4                # reference video (reference-based dimensions)
      ref.png                      # first frame (reference-free prompt_consistency)
      baseline.json
      checksums.sha256             # covers every file under reference/
      artifacts/                   # actions_file, action_path/action.npy, qa_dataset, .npz ...
  rubrics/<SceneName>/<provider>_<model>_<version>.json   # committed prompt-consistency rubrics
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

# Subset of dimensions, resumable
python tools/run_wmeval_lite.py run \
  --suite eval_sets/wmeval_lite_v1 \
  --model-root runs/genie3 \
  --dimensions causal_lpips,controllability \
  --resume \
  --output-dir reports/wmeval_lite/genie3
```

Candidate input is either the directory layout above (model name = directory
name; optional `model.yaml` for per-model params such as `shadow_hud_preset`)
or a strict JSONL manifest:

```json
{"model":"genie3","case_id":"MirrorRollingBall/third_collision","seed":"0001","candidate_video":"/path/to/rollout.mp4"}
{"model":"genie3","case_id":"RacingCar/first","seed":"0001","candidate_video":"/path/to/other.mp4","params":{"shadow_hud_preset":"genie"}}
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
  (seeds within a case, then across cases) and the suite-declared headline.
- `aggregate_summary.csv` — rows = models, columns = dimension headlines.

### Authoring cases

Reference files in place → `python tools/build_lite_case.py <case_dir>` writes
`checksums.sha256` and fills `evaluator_lock` (git revision + fingerprint).
Prompt-consistency rubrics: `python tools/run_wmeval_lite.py generate-rubrics`
(requires `OPENAI_API_KEY`; rubrics are committed, so this is an authoring
step, not part of evaluation).
