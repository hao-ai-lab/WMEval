# Evaluation Sets

Curated, Unity-free benchmark bundles. Reference Unity videos and artifacts ship
with the repo so evaluators never need a local Unity checkout.

## WMEval Lite v1

Path: `eval_sets/wmeval_lite_v1/`

One suite, one CLI, every dimension. `suite.yaml` declares the dimensions;
each case ships as a checksummed bundle under `cases/<scene>/<variant>/`.
Cases are **shared** across dimensions where it makes sense (e.g. the same
third-person scenes serve both `causal_lpips` and `controllability`).

| Dimension | Evaluator | Needs reference video | Cases |
| --- | --- | --- | --- |
| `causal_lpips` | registry (`lpips`, causal VLM collision) | yes | mirror_ball, ball_roller, beam_domino (third_person) |
| `controllability` | registry (`control_motion`) | yes (+ `action.npy`) | mirror_ball, ball_roller, beam_domino (third_person) |
| `prompt_consistency` | vlm_consistency judge | no (first frame only) | mirror_rolling_ball, racing_car (first/third) |
| `shadow_consistency` | registry (`shadow_consistency`) | no | reference-free scenes |
| `causal_step` | registry (`causal_step`) | yes (+ QA dataset) | beamdomino variants |
| `liquid` | registry (4 liquid modules) | yes (+ `.npz` artifacts) | liquid_container variants |

(Dimensions appear in `suite.yaml` as their data lands; run `list-cases` for
the current truth.)

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
{"model":"genie3","case_id":"mirror_ball/third_person","seed":"0001","candidate_video":"/path/to/rollout.mp4"}
{"model":"genie3","case_id":"racing_car/first","seed":"0001","candidate_video":"/path/to/other.mp4","params":{"shadow_hud_preset":"genie"}}
```

Legacy ids from the pre-suite manifests (e.g. `sample_scene`,
`mirror_ball_third_person`) are recorded in `suite.yaml` under
`legacy_case_aliases`; convert old candidate files with
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
