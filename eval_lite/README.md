# eval_lite — Prompt-Consistency (vlm_consistency) bundle

Self-contained, one-command automated evaluation for the **prompt-consistency**
dimension (`evaluation/modules/vlm_consistency`) over a small set of scenes.

Unlike `eval_sets/` (causal LPIPS / controllability, which compare a candidate
video against a reference *video*), prompt-consistency does **not** use a reference
video. It works like this:

```
prompt text ──> (LLM) ──> rubric (env + object checklist)
                                   │
candidate video ──> sampled frames ┤──> (VLM judge) ──> overall_score ∈ [0,1]
ref first-frame image ─────────────┘   (optional, shown as `ref_frame`)
```

Here the **Unity groundtruth video is itself the candidate being scored** (a
high-score sanity baseline). Provider is **OpenAI**.

## Layout

```
eval_lite/
  manifest.jsonl                      # THE source of truth (committed, hand/Claude-edited)
  data/<scene>/<video>.mp4            # the groundtruth candidate video (scored)
  data/<scene>/ref.png                # ref first-frame image (shown to the VLM)
  rubrics/<scene>/openai_<model>_v1.json   # pre-generated rubric (committed)
  run_eval_lite.sh                    # one-command runner
  README.md
```

This bundle is **manifest-driven**, like `eval_sets/`: `manifest.jsonl` is the
single committed source of truth — there is **no `scenes.csv`** and no build step.
Each line is one scene (one JSON object); the runner reads it directly.

Outputs go to `reports/eval_lite/results.jsonl` (the repo's `reports/` is
git-ignored, so results are not committed).

## manifest.jsonl format

One JSON object **per line** (JSONL — no surrounding array, no pretty-printing).
Paths are relative to `eval_lite/`. Required keys:

| key | meaning |
| --- | --- |
| `scene_name` | unique scene id, also the `data/<scene>/` folder name |
| `prompt` | text the rubric is generated from |
| `ref_path` | ref first-frame image, e.g. `data/<scene>/ref.png` (`null` to skip) |
| `video_path` | the candidate video to score, e.g. `data/<scene>/<video>.mp4` |
| `model_name` | label for the producer, e.g. `Unity` (free-form; shown in results) |
| `run_id` | label for the run, e.g. `gt` or `""` (free-form) |

Example line:

```json
{"scene_name":"RacingCar_third","prompt":"A very wide, straight highway ...","ref_path":"data/RacingCar_third/ref.png","video_path":"data/RacingCar_third/Unity_RacingCar_third.mp4","model_name":"Unity","run_id":""}
```

Because `model_name`/`run_id` are explicit fields, the **video filename can be
anything** — it no longer needs to contain the scene name.

## Adding a scene

1. Drop the scene's data under `data/<scene>/` (the video and `ref.png` at minimum).
2. Append one JSON line to `manifest.jsonl` (see format above). Ask Claude to do
   this if you'd rather not hand-edit JSON.
3. Generate the rubric + score it (rubrics for existing scenes are reused from cache):
   ```bash
   bash eval_lite/run_eval_lite.sh
   ```
4. Commit `data/<scene>/`, the updated `manifest.jsonl`, and the new
   `rubrics/<scene>/...json`.

## What you provide (per scene)

1. **Unity groundtruth video** → `data/<scene>/<video>.mp4`
2. **ref first-frame image** → `data/<scene>/ref.png`
3. **prompt text** → the `prompt` field in the manifest line
4. **`OPENAI_API_KEY`** — see [Set your OpenAI key](#set-your-openai-key) below.

## Set your OpenAI key

Both rubric generation and judging call OpenAI, so a key is required. The runner
resolves it in this order: **existing `OPENAI_API_KEY` env var**, then a **`.env`
file at the repo root**. Recommended (one-time) setup:

```bash
cp .env.example .env          # from the repo root
# edit .env and paste your key:  OPENAI_API_KEY=sk-...
```

`.env` is git-ignored, so your key is never committed. Alternatively,
`export OPENAI_API_KEY=sk-...` in your shell. If neither is set, the runner exits
with a clear message.

## Run

Use a Python environment that has the repo deps (`torch`, `cv2`, `pydantic`,
`numpy`) — e.g. the `ray_env` conda env. Point the runner at it with `PYTHON=...`
if your default `python` lacks them.

```bash
# normal run (reuses committed rubrics if present)
bash eval_lite/run_eval_lite.sh
# or, selecting the interpreter explicitly:
PYTHON=/Users/<you>/anaconda3/envs/ray_env/bin/python bash eval_lite/run_eval_lite.sh

# regenerate rubrics and overwrite the committed ones
OVERWRITE_RUBRICS=1 bash eval_lite/run_eval_lite.sh

# override model / weighting
MODEL=gpt-5.4 ALPHA=0.6 bash eval_lite/run_eval_lite.sh
```

Defaults: `MODEL=gpt-4.1`, `PROMPT_VERSION=v1`, `ALPHA=0.6` (environment weight).

## Output

`reports/eval_lite/results.jsonl` — one row per video, e.g.:

```json
{"scene_name":"mirror_ball","model_name":"Unity","run_id":"gt","status":"ok",
 "environment_result":{"score":0.83,"item_scores":[...],"explanations":{...}},
 "object_result":{"score":0.92,"item_scores":[...],"explanations":{...}},
 "overall_score":0.866}
```

`overall_score = alpha * environment_score + (1 - alpha) * object_score`.

## How it works under the hood

`run_eval_lite.sh` just drives the existing, tested pipeline
(`tools/evaluate_world_model_vlm_consistency.py` → `batch.py`) pointed at this
folder: `generate_rubrics` → `evaluate`, both reading `manifest.jsonl` directly.
(`batch.py` also has a `build_manifest` subcommand that derives a manifest from a
`scenes.csv`; this bundle skips it and maintains the manifest by hand.) No module
code is specific to `eval_lite/`.
