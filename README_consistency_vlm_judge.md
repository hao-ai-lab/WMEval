# VLM Consistency: Dual-Agent Rubric for Video–Prompt Alignment

> A reference implementation of a dual-agent, rubric-driven VLM judge for evaluating
> how faithfully a generated video reflects its text prompt. Rubrics are split by
> semantic dimension (scene-level *environment* vs. salient *objects*); a single
> VLM call then scores all timestamped frames against every rubric.

---

## 1. Overview

Evaluating text-to-video generation with a vision–language model is sensitive to
*what* the judge is asked. A free-form "rate this video" prompt produces
inconsistent scores; a flat checklist cannot distinguish scene-level fidelity
from per-object fidelity, and tends to over-penalise natural temporal dynamics.

We address this with a **two-stage pipeline**:

1. **Dual-agent rubric generation.** Two independent LLM agents produce binary
   yes/no criteria for two disjoint semantic scopes — *environment* and *object*
   — given the prompt and (optionally) a ground-truth reference frame.
2. **Unified VLM judge.** All uniformly sampled, timestamp-labelled frames are
   passed in a single VLM call; per-rubric chain-of-thought is enforced via
   a strict-mode JSON schema (`reasoning -> frame_evidence -> judged_score`).

A weighted hierarchical score is computed:

```
EnvironmentScore = sum(w_i * s_i) / (3 * sum(w_i))    over environment rubrics
ObjectScore      = sum(w_i * s_i) / (3 * sum(w_i))    over object rubrics
OverallScore     = alpha * EnvironmentScore + (1 - alpha) * ObjectScore
```

with `s_i in {0,1,2,3}`, `w_i in {1.0, 1.5, 2.0}`, default `alpha = 0.6`.

---

## 2. Pipeline

```
                  +---------------------------+
                  |         Prompt            |
                  | (+ optional GT frame img) |
                  +-------------+-------------+
                                |
              +-----------------+-----------------+
              |                                   |
              v                                   v
     +----------------+                  +----------------+
     |  Environment   |                  |     Object     |
     |  Agent (LLM)   |                  |  Agent (LLM)   |
     |  0..4 rubrics  |                  |  0..4 rubrics  |
     +-------+--------+                  +--------+-------+
             |                                    |
             +------------------+-----------------+
                                |
                                v
                      cap to 5 per agent (sorted by weight)
                      + inject mandatory holistic coverage
                        rubric per agent (env_00 / obj_00, w=2.0)
                                |
                                v
                       Rubric JSON cache
              rubrics/{scene}/{provider}_{model}_{ver}.json
                                |
                                v
   +----------------------------+----------------------------+
   |  Generated video                                        |
   |   uniform sample N frames over T seconds                |
   |   labels: frame_t=X.Xs (+ optional gt_frame_t=X.Xs)     |
   +----------------------------+----------------------------+
                                |
                                v
                  +----------------------------+
                  |   Unified VLM Judge call   |
                  |   strict-mode JSON schema  |
                  |   reasoning -> evidence    |
                  |     -> judged_score (0-3)  |
                  +-------------+--------------+
                                |
                                v
              EnvironmentScore + ObjectScore -> Overall
```

---

## 3. Method

### 3.1 Dual-agent rubric generation

Two specialised system prompts elicit *atomic* binary criteria within disjoint
scopes:

- **Environment agent** (`env_*`) targets *scene-level* attributes only —
  structure, layout, lighting, sky, terrain, style.
- **Object agent** (`obj_*`) targets *salient entities* — characters, vehicles,
  primary props — and their attributes (color, material, texture, shape).

Each agent is asked to cover at most two sub-criteria per rubric:
(a) **prompt-fidelity** in early frames, (b) **late-frame drift**. Drift is
evaluated under an OR rule: a late frame passes if it is *either* consistent
with early frames *or* still aligned with the prompt; only joint violation
fails. This prevents penalising natural motion or camera rotation.

Two **mandatory holistic rubrics** (`env_00`, `obj_00`, weight 2.0) are
auto-injected to ask whether *all* environment elements / characters mentioned
in the prompt appear at any point in the video. They are not subject to LLM
generation.

After capping to `MAX_PER_AGENT = 5`, every scene yields up to **6 rubrics per
agent** (5 generated + 1 mandatory).

Reference grounding is optional: if a GT initial frame is provided, agents may
emit `reference_grounded` rubrics for *major* visual anchors visible in the
reference; otherwise all rubrics are forced to `prompt_grounded`.

### 3.2 Frame sampling

Candidate frames are sampled uniformly across the first `T` seconds of the
video and labelled with their actual timestamps (`frame_t=X.Xs`). The default
is `T = 10s`, `N = 18` frames (1.8 fps). When a GT reference video is
available, its frames are passed alongside as `gt_frame_t=X.Xs`.

### 3.3 Unified VLM judge

A single VLM request carries all candidate frames, the rubric set, and a
strict-mode JSON schema enforcing chain-of-thought field order. The judge is
explicitly instructed to:

- ignore natural temporal dynamics unrelated to a rubric;
- use OR logic for late-frame drift questions;
- produce evidence (`frame_evidence`) before assigning a score.

The `judged_score` is a 4-level ordinal:

| score | meaning                                                       |
|-------|---------------------------------------------------------------|
| 3     | criterion fully met; no meaningful deviation                  |
| 2     | met in vast majority; only minor flicker/transient occlusion  |
| 1     | partially met; satisfied in some frames, wrong in others      |
| 0     | not met / contradicted, or majority of frames corrupted       |

### 3.4 Aggregation

Per-agent score is the weight-normalised mean of `judged_score / 3`. The
overall score combines them with `alpha` (default `0.6` favouring
environment). If one agent has zero rubrics, its component is excluded.

---

## 4. Repository Structure

```
vlm_consistency/
  __init__.py            public API: Rubric, JudgeResult, generate_rubric, judge_scene, ...
  __main__.py            python -m vlm_consistency entry point
  run_pipeline.py        single-scene CLI (generate_rubrics / evaluate)
  batch.py               batch CLI across model directories
  rubric_generator.py    dual-agent rubric generation
  vlm_judge.py           unified VLM judging + score aggregation
  rubric_cache.py        on-disk rubric cache (save/load/find)
  frame_sampler.py       uniform timestamped frame sampling
  schemas.py             Pydantic models + provider-specific schema transforms
  providers/
    openai.py            OpenAI Responses API (structured output)
    anthropic.py         Anthropic Messages API (tool-use schema)
    gemini.py            Google Gemini structured output
```

Test assets (videos, prompts, reference frames) live outside the module under
`WMEval/tests/vlm_judge/`.

---

## 5. Installation

The module depends on Python 3.10+ and the following libraries:

```
pip install pydantic opencv-python numpy openai anthropic google-genai
```

Provide an API key for the provider(s) you intend to use:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

---

## 6. Data Preparation

The default batch layout (overridable via CLI flags) is:

```
WMEval/tests/vlm_judge/
  Prompt - Basic_env.csv          # columns: Scene, Prompt
  data/
    Genie/   *.mp4                # one subdir per generator model
    Matrix/  *.mp4
    Unity/   *.mp4
  ref_img/
    {SceneName}_image.png         # GT initial frames per scene
```

**Video-to-scene matching** (in `batch.py`): a `.mp4` is associated with a
scene when, after stripping the model-name prefix and any
`_firstperson` / `_thirdperson` / `_actionsrun` / `_long` suffix, the
remainder is a prefix of the normalised scene name.

**CSV schema.** Two columns are required:

| column     | description                                |
|------------|--------------------------------------------|
| `Scene`    | scene identifier (used as cache key)       |
| `Prompt`   | natural-language scene description         |

---

## 7. Usage

All commands assume `cd WMEval` so that the default `tests/vlm_judge/...` paths
resolve.

### 7.1 Quick start (batch)

```bash
# (1) Generate rubrics once per (provider, model)
python -m evaluation.modules.vlm_consistency.batch generate_rubrics \
    --provider openai --model gpt-5.4 \
    --rubric_dir rubrics --prompt_version v1

# (2) Score every video under tests/vlm_judge/data/* against the rubrics
python -m evaluation.modules.vlm_consistency.batch evaluate \
    --provider openai --model gpt-5.4 \
    --rubric_dir rubrics --prompt_version v1 \
    --alpha 0.6 \
    --output batch_result.json
```

### 7.2 Useful variants

```bash
# Limit to specific generator model directories
python -m evaluation.modules.vlm_consistency.batch evaluate \
    --provider openai --model gpt-5.4 \
    --model_names Genie Unity \
    --output batch_genie_unity.json

# Disable GT reference image
python -m evaluation.modules.vlm_consistency.batch evaluate \
    --provider openai --model gpt-5.4 --no_ref \
    --output batch_noref.json

# Decouple judge model from rubric model (rubric cache is reused)
python -m evaluation.modules.vlm_consistency.batch evaluate \
    --provider anthropic --model claude-opus-4-7 \
    --rubric_provider openai --rubric_model gpt-5.4 \
    --output batch_judge_anthropic.json

# Custom paths
python -m evaluation.modules.vlm_consistency.batch evaluate \
    --provider openai --model gpt-5.4 \
    --csv path/to/prompts.csv \
    --data_dir path/to/videos --ref_dir path/to/refs \
    --output result.json
```

### 7.3 Single-scene CLI

For ad-hoc evaluation of one video against one cached rubric:

```bash
python -m evaluation.modules.vlm_consistency evaluate \
    --video  tests/vlm_judge/data/Unity/Unity_BallRoller.mp4 \
    --rubric rubrics/BallRoller/openai_gpt-5-4_v1.json \
    --provider openai --model gpt-5.4 \
    --gt_frame tests/vlm_judge/ref_img/BallRoller_image.png \
    --alpha 0.6 --duration 10 --num_frames 18 \
    --output results/Unity_BallRoller.json
```

### 7.4 Programmatic API

```python
from evaluation.modules.vlm_consistency import (
    generate_rubric, save_rubric, load_rubric, find_rubric,
    judge_scene,
)
from evaluation.modules.vlm_consistency.frame_sampler import prepare_frames_for_vlm

rubric = generate_rubric(
    prompt_text="A red ball rolling on a chessboard ground...",
    scene_name="ChessBoardwithSphere",
    provider="openai",
    model="gpt-5.4",
    gt_frame_path="tests/vlm_judge/ref_img/ChessBoard_image.png",
)
save_rubric(rubric, rubric_dir="rubrics", prompt_version="v1")

frames, n = prepare_frames_for_vlm(
    video_path="tests/vlm_judge/data/Unity/Unity_ChessBoard.mp4",
    first_n_seconds=10.0, sample_fps=1.8, max_frames=18,
    use_timestamp_labels=True,
)
result = judge_scene(rubric=rubric, all_frames=frames,
                    provider="openai", model="gpt-5.4", alpha=0.6)
print(result.overall_score, result.environment_result.score, result.object_result.score)
```

---

## 8. Output Format

Each batch entry is a `JudgeResult` plus three bookkeeping fields:

```jsonc
{
  "scene_name": "ChessBoardwithSphere",
  "provider": "openai",
  "model": "gpt-5.4",
  "rubric_metadata": {
    "generator_provider": "openai",
    "generator_model": "gpt-5.4",
    "generated_at": "2025-12-01T08:32:11+00:00",
    "num_candidates_generated": 7
  },
  "alpha": 0.6,
  "num_frames": 18,
  "environment_result": {
    "agent": "environment",
    "item_scores": [
      {
        "rubric_id": "env_01",
        "reasoning": "Frames 0.5s-9.5s show a black-and-white grid floor...",
        "frame_evidence": ["frame_t=0.5s", "frame_t=4.5s", "frame_t=9.5s"],
        "judged_score": 3
      }
    ],
    "score": 0.8889,
    "explanations": { "summary": "...", "issues": ["..."] }
  },
  "object_result": { /* same shape with obj_* rubrics */ },
  "overall_score": 0.8333,
  "model": "Unity",         // batch only: generator-model directory name
  "video_file": "Unity_ChessBoard.mp4",
  "status": "ok"
}
```

`status` may also be `missing_rubric`, `missing_video`, or `error`.

---

## 9. Defaults and Hyperparameters

| symbol         | default | meaning                                     |
|----------------|---------|---------------------------------------------|
| `alpha`        | 0.6     | weight on EnvironmentScore in overall       |
| `T` / duration | 10.0 s  | window from start of video                  |
| `N` / frames   | 18      | uniformly sampled candidate frames          |
| weights        | {1, 1.5, 2} | rubric weight quantisation              |
| max per agent  | 5 (+1)  | LLM-generated cap + mandatory holistic      |
| score scale    | 0..3    | per-rubric ordinal, normalised to [0,1]     |

---

## 10. Caching

Rubrics are cached on disk by `(scene, provider, model, prompt_version)`:

```
rubrics/
  {SceneName}/
    {provider}_{model}_{version}.json
```

Names are sanitised (dots and other punctuation become `-`) so file names stay
portable across OSes. `find_rubric(...)` performs a substring match on the
sanitised stem so partial keys (e.g. only `prompt_version`) still resolve.

The judge call always re-loads from this cache; rubric and judge models are
therefore **decoupled** — pre-compute rubrics once with a strong LLM, then
sweep cheaper VLMs as judges.

---

## 11. Supported Providers

| provider   | env var               | structured-output backend         |
|------------|-----------------------|-----------------------------------|
| `openai`   | `OPENAI_API_KEY`      | Responses API, strict JSON schema |
| `anthropic`| `ANTHROPIC_API_KEY`   | Messages API tool-use schema      |
| `gemini`   | `GEMINI_API_KEY`      | `responseSchema` with inlined `$ref` |

Provider-specific JSON-schema transforms (`schemas._transform_for_*`) strip
features each backend rejects (e.g., `min/maxItems` for OpenAI strict, `$defs`
for Gemini).

---

## 12. Limitations

- The pipeline judges *appearance fidelity*; it does **not** assess physics,
  task completion, or action quality.
- Rubric quality depends on the rubric LLM — pathological prompts can yield
  vacuous criteria. We recommend the strongest available model for rubric
  generation.
- The mandatory holistic rubrics (`*_00`) are coarse coverage checks, not
  spatial-grounding; they are intentionally lenient.
- Single-call judging trades fine-grained per-frame attention for cost and
  consistency. For long-horizon videos, increase `--num_frames` and
  `--duration`, or split the video into segments.

---

## 13. Reproducibility

- **Determinism.** Provider sampling temperatures should be fixed at 0 (or the
  lowest the API permits). Outputs still vary across model snapshots — pin
  exact model IDs in `--model`.
- **Seeded sampling.** Frame sampling is deterministic given `(T, N,
  start_seconds)`.
- **Cache invariance.** When changing rubric prompts, bump `--prompt_version`
  to keep old rubrics intact for ablations.

---

