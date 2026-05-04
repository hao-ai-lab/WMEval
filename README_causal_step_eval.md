# CausalStep Evaluation README

This document explains how to run the `causal_step` evaluation in the current
`WMEval` framework end-to-end: from a Unity BeamDomino physics log through
segment detection, QA generation, human review, and finally the stepwise
VLM-judge scoring that compares a world-model video against its Unity ground
truth.

The module adapts the **CausalStep** protocol (arXiv 2507.16878) to a pairwise
world-model vs. Unity-reference setting: the same QA dataset is graded twice
(once on the Unity GT video, once on the world-model candidate) with a shared
judge, and the per-metric `{unity, wm, delta}` block is reported.

## 1) Pipeline overview

Three CLIs hand data to each other through files. They all run in the WMEval
venv:

```
tools/detect_beamdomino_segments.py
    in:  unity_gt_log.json        (from GroundTruthRecorder.cs in Unity)
    out: segments.json            (4 events: rolling / impact / topple / cascade)

tools/generate_beamdomino_causal_step_qa.py
    in:  segments.json
    out: qa_dataset.json          (CausalStep schema; HUMAN-REVIEW REQUIRED)

tools/evaluate_world_model_causal_step.py
    in:  manifest.jsonl           (unity_video, wm_video, qa_dataset per line)
    out: causal_step_results.json (per_video + aggregate metrics)
```

The upstream Unity recording side (running the BeamDomino scene and writing
`unity_gt_log.json` + `unity_gt.mp4`) lives in a sibling repo
(`WorldModelBenchScenePrivate/GameScenes/BeamDomino/`) and uses a separate
`mlagents` conda env. This README is scoped to everything that happens on the
WMEval side once you have `unity_gt_log.json`, `unity_gt.mp4`, and your
world model's `wm_pred.mp4` on disk.

## 2) Scope

- Main entry: `tools/evaluate_world_model_causal_step.py`
- Supporting CLIs:
  - `tools/detect_beamdomino_segments.py`
  - `tools/generate_beamdomino_causal_step_qa.py`
- Batch mode only: JSONL manifest (`--manifest`), one trajectory per line.
- Two judges are supported via `--judge`:
  - `openai` — `gpt-4o` by default, requires `OPENAI_API_KEY`.
  - `fake_always_a` — debug judge that always returns `A`; used for plumbing
    checks and identity tests. Metrics will **not** reflect video content.

## 3) Environment

The module drives the vendored CausalStep package at
`imports/CausalStep/`. It has no `setup.py`; the runner injects it onto
`sys.path` via `evaluation/modules/_causal_step_io.py::ensure_causalstep_importable`.
Keep it at that path or pass `--causalstep-root /custom/path`.

Create a dedicated WMEval venv (do **not** reuse the Unity `mlagents` env —
its pinned `numpy`/`protobuf` versions fight `opencv-python` and `openai`):

```bash
cd WMEval
python -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy openai pytest
export OPENAI_API_KEY=sk-...   # only needed for the real (gpt-4o) judge
```

Verify CausalStep is importable from the venv:

```bash
pytest tests/test_causal_step_delta.py::test_causalstep_package_importable -v
```

If it fails, either `CausalStep/` is not at `imports/CausalStep/` or the
`causalstep/` Python package inside it is missing files.

## 4) Step 1 — Detect segment boundaries

`tools/detect_beamdomino_segments.py` parses the per-frame physics log and
classifies the trajectory into one of three **shapes**, each emitting its own
chain of segments. Each segment is bounded to capture the *visible dynamic
phase* its QA template depicts, so the evenly-sampled frames the judge sees
are dominated by motion rather than stillness.

The classifier looks at two things from the GT log:

- **what the FloorBall hits first** after motion onset (a `Pillar`/`Roof` of
  the platform, or a `Domino` directly), and
- **whether the `Roof` body ever tilts past `--platform-tilt-threshold-deg`**.

| `trajectory_shape` | Condition | Stage chain (segment count) |
| --- | --- | --- |
| `canonical`     | Roof tilt crosses the threshold | `rolling` → `impact` → `topple` → `cascade` (4) |
| `glancing`      | First contact is `Pillar`/`Roof` but Roof never crosses threshold | `rolling` → `glance` → `strike` → `cascade_from_strike` (4) |
| `direct_strike` | First non-ground FloorBall contact is a `Domino` (ball bypasses platform) | `rolling` → `strike` → `cascade_from_strike` (3) |

Boundary rules per stage:

| Stage | Boundary rule |
| --- | --- |
| `rolling` | FloorBall speed first crosses `--motion-onset-velocity-mps` (default `0.5 m/s`) until the first qualifying `"Ball"` collision (impulse ≥ `--ball-impulse-threshold`, default `0.01`; ignored before `--ball-impact-min-time-sec`). |
| `impact` (canonical) | ball-`Pillar` contact until `Roof` body tilt first crosses `--platform-tilt-threshold-deg` (default `5°`). If the threshold isn't crossed before the first fallen domino, the boundary collapses to the midpoint between ball impact and first fallen domino. |
| `topple` (canonical) | platform-tilt onset until the first frame where any `Domino_*` body has tilted past `--fallen-tilt-deg` (default `30°`, computed as `max(|wrap180(rot.x)|, |wrap180(rot.z)|)`). |
| `glance` (glancing) | first ball-`Pillar`/`Roof` contact until first ball-`Domino` contact. |
| `strike` (glancing/direct_strike) | first ball-`Domino` contact until first fallen domino. |
| `cascade` / `cascade_from_strike` | first fallen domino until the chain plateaus (no new fall for `--plateau-window-sec`, default `0.5s`). |

The two cascade stages share descriptive QAs (the chain reaction looks the
same regardless of cause) but have different *causal* answers — `cascade`
attributes the first fall to the falling platform, `cascade_from_strike`
attributes it to the direct ball-on-domino strike.

The detector falls back gracefully when the log is minimal:
`bodies[].velocity` absent → `rolling` starts at `t=0`; no `Roof` body and no
`Pillar` contact in the scene at all → shape is forced to `canonical` so old
fixtures (e.g. `tiny_log.json`) keep parsing.

```bash
python tools/detect_beamdomino_segments.py \
  --log   runs/traj_001/unity_gt_log.json \
  --out   runs/traj_001/segments.json
```

For the trajectories driven by `controller.py` (`clean`/`far`/`glancing`/`miss`),
Unity reports `FloorBall<->Pillar` / `FloorBall<->Domino_*` contacts with
`impulse=0.0` (the structure-borne `RoofBall<->Roof` shake that follows
carries the energy). The default `--ball-impulse-threshold 0.01` rejects
those zero-impulse contacts, so detection fails. Use:

```bash
python tools/detect_beamdomino_segments.py \
  --log   runs/traj_<name>/unity_gt_log.json \
  --out   runs/traj_<name>/segments.json \
  --ball-impulse-threshold   0.0 \
  --ball-impact-min-time-sec 1.0
```

`--ball-impact-min-time-sec` skips the t≈0.02s settling contacts so the
zero-impulse threshold doesn't latch onto them.

Optional tuning flags (rarely needed; defaults are tuned to the BeamDomino
scene): `--fallen-tilt-deg`, `--ball-impulse-threshold`, `--plateau-window-sec`,
`--ball-impact-min-time-sec`, `--motion-onset-velocity-mps`,
`--platform-tilt-threshold-deg`.

The output `events[].trigger` names the boundary that fired:
`motion_onset` / `ball_impulse` / `ball_pillar` / `ball_domino` /
`platform_tilt` / `first_fallen`. The top-level `trajectory_shape` field
names the dispatched chain and is consumed by the QA generator (next
section).

If the detector raises `SegmentDetectionError`, either no `"Ball"` body ever
collided with sufficient impulse (`no ball impact found ...`) or no domino
ever tilted past the threshold (`no fallen domino found ...`). The
trajectory probably never propagated to a domino fall — re-record with a
stronger ball push, or lower `--ball-impulse-threshold` / `--fallen-tilt-deg`
if the defaults don't match your scene's physics.

## 5) Step 2 — Generate a draft QA dataset (then review it)

`tools/generate_beamdomino_causal_step_qa.py` turns each segment into a
CausalStep `descriptive` + `causal` QA pair. Two modes are supported.

If `segments.json` carries a `trajectory_shape` field (the detector emits
one), the generator picks templates from `SHAPE_STAGE_CHAINS[shape]` rather
than from each event's `name`. This decouples the QA semantics from the
detector's segment labels — the same `segments.json` always produces the
shape-correct chain. In particular: `direct_strike` produces 3 segments
(rolling/strike/cascade_from_strike), `glancing` produces 4 segments
(rolling/glance/strike/cascade_from_strike) with a "platform survives" middle
phase, and `canonical` keeps the original 4 segments
(rolling/impact/topple/cascade). When `trajectory_shape` is absent (legacy
fixtures), each event's `name` field selects the template directly.

In both modes, choice positions are **auto-shuffled** per QA via a
deterministic seed `(video_id, segment_index, qa_type)` — `correct_answer`
varies across A/B/C/D rather than always being `A`. Reruns on the same
inputs produce identical output (the shuffle is stable), so the OpenAI
judge's prompt-based cache key stays valid.

The output is a **draft** in either mode and must be reviewed by a human.

### Mode 1 — Templates (default; offline, free)

Static per-stage templates ground the wording in the canonical BeamDomino
scene (one ball rolls along the floor → strikes a leg of a small white
platform with another ball on top → the platform tips → a row of dominoes
falls in a chain). Distractors are written as plausible near-misses (mirror
direction, wrong contact location, simultaneous-vs-sequential fall) rather
than easy rejects like "wind" or "magic", so the VLM judge has to look at
the frames.

```bash
python tools/generate_beamdomino_causal_step_qa.py \
  --segments  runs/traj_001/segments.json \
  --video-id  traj_001 \
  --out       runs/traj_001/qa_dataset.json
```

### Mode 2 — VLM hybrid (online; OpenAI API)

Asks a vision model to rewrite the question and generate 3 distractors
*grounded in the segment's actual frames* — useful when trajectories vary
visibly (e.g., which side the leg gets knocked from, which arc of dominoes
falls) and a static template can't reflect that.

**Hybrid contract:** the template's correct answer is given to the VLM as
a strong prior. The VLM may either confirm it (return verbatim) or *correct*
it when the segment frames clearly contradict the template wording — for
example, when the template says "the platform tipped right" but the frames
show it tipping left. Every override is logged to stderr (`[qa-gen] VLM
overrode template correct answer ...`) so a human reviewer can spot-check
the change before evaluation. This trades the previous "VLM cannot corrupt
ground truth" guarantee for the ability to fix templates that are wrong
for a particular trajectory's footage. If you want the strict-pin
behaviour, run in templates mode instead.

```bash
export OPENAI_API_KEY=sk-...
python tools/generate_beamdomino_causal_step_qa.py \
  --segments  runs/traj_001/segments.json \
  --video-id  traj_001 \
  --out       runs/traj_001/qa_dataset.json \
  --mode      vlm \
  --video     runs/traj_001/unity_gt.mp4 \
  --vlm-model gpt-5.4
```

VLM responses are disk-cached at `<out>.parent/qa_vlm_cache/` (override with
`--vlm-cache-dir`). Cache keys include `--vlm-num-frames` (default `4`) so
bumping the frame count forces a recompute.

### Human review

Open `qa_dataset.json` and edit:

- Reword anything too generic to discriminate WM from GT.
- For VLM mode, scan the QA-gen stderr for `VLM overrode template correct
  answer ...` lines and confirm every override matches what the video
  actually shows — the VLM is now allowed to change ground truth, so this
  is the most important review step.
- Double-check that no distractor accidentally restates the correct answer
  — `decorate_qa` rejects exact duplicates but not paraphrases.
- Verify segment time spans look sensible against the actual video.
- The `correct_answer` letter is auto-correct after the shuffle; if you
  reorder choices manually, update the letter.

## 6) Step 3 — Smoke-test with the fake judge

Run this first when changing anything in the pipeline — no API key, no cost.
The fake judge returns `"A"` for every question, so it exercises every code
path without hitting OpenAI.

```bash
pytest tests/test_causal_step_delta.py -v
pytest tests/test_beamdomino_segments.py -v
pytest tests/test_causal_step_module.py -v
```

All three suites should pass before doing a real run.

## 7) Step 4 — Build a manifest

The evaluator iterates a JSONL manifest, one entry per line. Four keys are
required, one is optional:

| Key | Required | Meaning |
| --- | --- | --- |
| `video_id`    | yes | Stable id used for cache dirs and per-video output rows. Must match a `video_id` inside the QA dataset. |
| `unity_video` | yes | Path to the Unity ground-truth video (reference side). |
| `wm_video`    | yes | Path to the world-model candidate video. |
| `qa_dataset`  | yes | Path to the CausalStep-schema QA JSON for this trajectory. |
| `trajectory`  | no  | Free-form group label (e.g. `clean`, `far`). When present, the aggregator emits a `by_trajectory.<label>` block with mean+std per metric across the entries that share that label. |

```jsonl
{"video_id":"traj_001","unity_video":"runs/traj_001/unity_gt.mp4","wm_video":"runs/traj_001/wm_pred.mp4","qa_dataset":"runs/traj_001/qa_dataset.json"}
{"video_id":"traj_002","unity_video":"runs/traj_002/unity_gt.mp4","wm_video":"runs/traj_002/wm_pred.mp4","qa_dataset":"runs/traj_002/qa_dataset.json"}
```

Paths can be absolute or relative to the directory you launch the command
from — be consistent across a batch. Per-entry failures are caught and
reported as `{"video_id": ..., "error": "..."}` in `per_video` so one bad
entry does not halt the batch.

### Multi-run pattern (one Unity reference, N world-model rollouts)

When the world model is non-deterministic (e.g. Project Genie sampled 5
runs per trajectory), use one manifest entry per run, all sharing the same
Unity video and QA dataset. Each entry needs a distinct `video_id` because
the per-video answer cache is keyed by it; the QA file must list every
id under `videos[]`.

```jsonl
{"video_id":"traj_clean_run1","trajectory":"clean","unity_video":"runs/traj_clean/Unity_clean.mp4","wm_video":"runs/traj_clean/genie/run1.mp4","qa_dataset":"runs/traj_clean/qa_dataset.json"}
{"video_id":"traj_clean_run2","trajectory":"clean","unity_video":"runs/traj_clean/Unity_clean.mp4","wm_video":"runs/traj_clean/genie/run2.mp4","qa_dataset":"runs/traj_clean/qa_dataset.json"}
…
```

The QA generator emits one `videos[]` entry under `--video-id`; for the
multi-run pattern, post-process the JSON to duplicate that entry across
`run1`..`runN` ids (the QA content is identical, only the id changes).

### Asymmetric judging (`--wm-full-video`)

When the WM frames aren't time-aligned to the Unity reference (Genie's
manually-driven rollouts produce wildly different durations and per-run
event timings, see §19), Unity-derived segment windows slice the wrong
frames out of the WM clip. Pass `--wm-full-video` to skip windowing on
the WM side only — the judge sees the whole clip for every QA on WM,
while the Unity side keeps using segment windows. The chain/restart
metrics still compute correctly because CausalStep's `PrecomputedModel`
doesn't care how the answers were obtained; the trade-off is that the
"stepwise" framing is weakened on the WM side (the judge can localise
events anywhere in the full clip rather than within a specific segment).

## 8) Step 5 — Dry run with the fake judge on real videos

```bash
python tools/evaluate_world_model_causal_step.py \
  --manifest   runs/manifest.jsonl \
  --cache-root runs/cache \
  --judge      fake_always_a \
  --out        runs/causal_step_results_dry.json
```

This confirms videos load (cv2 reads frames + reports a real fps; both
Unity 60fps and Genie 20fps clips are auto-detected per file via
`cv2.CAP_PROP_FPS`), segments fire, and the delta block aggregates — all
for free.

## 9) Step 6 — Real run

```bash
export OPENAI_API_KEY=sk-...
python tools/evaluate_world_model_causal_step.py \
  --manifest   runs/manifest.jsonl \
  --cache-root runs/cache \
  --judge      openai \
  --out        runs/causal_step_results.json
```

Useful flags (see `--help` for the full list):

| Flag | Purpose |
| --- | --- |
| `--wm-full-video` | Skip segment-windowing on the WM side (see §7). |
| `--vlm-model gpt-5.4-mini` | Override the OpenAI model id. Default is the module's `causal_step_vlm_model` (currently `gpt-5.4-mini`). Cache keys include the model id, so switching here doesn't collide with prior cached answers. |
| `--vlm-num-frames N` | Number of evenly-spaced frames the judge sees per QA. Default `4`. Bumping to `8` materially helps on tight segments — for a `direct_strike` cascade window of 0.12s at 60fps (7 frames), 4 samples undersample the chain reaction. Cache keys include this number, so changing it forces a recompute. |
| `--causalstep-root /abs/path` | Point at a non-default `CausalStep/` checkout. |

Equivalent script file:

- `scripts/evaluate_world_model_causal_step.sh`

## 10) VLM answer cache

All judge responses are cached under:

```
<cache-root>/<video_id>/vlm_cache/
```

Re-runs with the same cache root are cheap on retries. If you change the
prompt version (artifact `causal_step_vlm_prompt_version`), invalidate the
cache by deleting that directory — cache keys do not currently include the
prompt version.

Per-video written answers (in CausalStep `PrecomputedModel` format) land at:

```
<cache-root>/<video_id>/{unity,wm}/
```

## 11) Output fields and interpretation

The tool writes a single JSON with two top-level keys:

```json
{
  "per_video": [
    {
      "module": "causal_step",
      "video_id": "traj_clean_run1",
      "trajectory": "clean",
      "unity":  { "CSR": ..., "AMCL": ..., "MCL": ..., "RF": ..., "WS": ..., "DUA": ..., "ICRA": ... },
      "wm":     { "CSR": ..., "AMCL": ..., "MCL": ..., "RF": ..., "WS": ..., "DUA": ..., "ICRA": ... },
      "delta":  { "csr": ..., "amcl": ..., "mcl": ..., "rf": ..., "ws": ..., "dua": ..., "icra": ... },
      "delta_convention": "positive = WM worse than Unity",
      "vlm_model": "gpt-5.4-mini",
      "judge_kind": "openai",
      "causal_fail_advance": true,
      "wm_full_video": true
    }
  ],
  "aggregate": {
    "unity_mean":  { "csr": ..., "..." : "..." },
    "wm_mean":     { "csr": ..., "..." : "..." },
    "delta_mean":  { "csr": ..., "..." : "..." },
    "n_trajectories": 15,
    "n_errors": 0,
    "delta_convention": "positive = WM worse than Unity",
    "by_trajectory": {
      "clean": {
        "n_runs": 5,
        "delta_mean": { "csr": ..., "..." : "..." },
        "delta_std":  { "csr": ..., "..." : "..." },
        "wm_mean":    { "csr": ..., "..." : "..." },
        "wm_std":     { "csr": ..., "..." : "..." },
        "unity_mean": { "csr": ..., "..." : "..." }
      },
      "far":      { "n_runs": 5, "..." : "..." },
      "glancing": { "n_runs": 5, "..." : "..." }
    }
  }
}
```

The same `aggregate` block is printed to stdout. `aggregate.delta_mean` is the
headline number per metric. `aggregate.by_trajectory` is only emitted when
manifest entries carry a `trajectory` field; the std blocks use population
stdev (zero for single-entry groups).

### Metric reference

Higher-is-better unless noted (see `evaluation/modules/_causal_step_delta.py`):

- `CSR` — Causal Success Rate (%)
- `AMCL` — Average Max Causal-chain Length
- `MCL` — Max Causal-chain Length
- `WS` — Weighted Score
- `DUA` — Descriptive Understanding Accuracy (%)
- `ICRA` — Isolated Causal Reasoning Accuracy (%)
- `RF` — Restart Frequency (**lower is better**; delta is sign-inverted so
  that `positive = WM worse` still holds).

### Interpretation

A `delta_mean.csr` of `+20.0` means the world-model video lost 20 percentage
points of CSR versus the Unity reference, averaged across trajectories.
Polarity is set deliberately in `causal_step.py` and `_causal_step_delta.py` —
for every metric, **positive means "WM is worse than Unity"**.

## 12) Typical iteration loop

1. Record one more Unity trajectory → `unity_gt_log.json` + `unity_gt.mp4`.
2. Generate `wm_pred.mp4` from your world model on the same initial conditions.
3. Run the three WMEval CLIs:
   ```bash
   python tools/detect_beamdomino_segments.py \
     --log .../unity_gt_log.json --out .../segments.json \
     --ball-impulse-threshold 0.0 --ball-impact-min-time-sec 1.0

   # Option A — templates (offline, free)
   python tools/generate_beamdomino_causal_step_qa.py --segments .../segments.json --video-id traj_N --out .../qa_dataset.json

   # Option B — VLM hybrid (frame-grounded distractors; needs OPENAI_API_KEY)
   # python tools/generate_beamdomino_causal_step_qa.py --segments .../segments.json --video-id traj_N --out .../qa_dataset.json \
   #     --mode vlm --video .../unity_gt.mp4 --vlm-model gpt-5.4

   $EDITOR .../qa_dataset.json   # human review
   echo '{"video_id":"traj_N","trajectory":"clean","unity_video":".../unity_gt.mp4","wm_video":".../wm_pred.mp4","qa_dataset":".../qa_dataset.json"}' >> runs/manifest.jsonl
   python tools/evaluate_world_model_causal_step.py --manifest runs/manifest.jsonl --cache-root runs/cache \
     --judge openai --vlm-model gpt-5.4-mini --vlm-num-frames 8 --wm-full-video \
     --out runs/causal_step_results.json
   ```
4. Inspect `aggregate.delta_mean` (overall) and `aggregate.by_trajectory.<label>.delta_mean ± delta_std` (per-group, when entries carry a `trajectory` field).

## 13) Causal-failure restart semantics

The CausalStep paper is internally inconsistent about what happens after an
incorrect causal answer:

- **Prose**: restart from the descriptive QA of the *same* segment.
- **Algorithm 1 pseudocode**: advance to the *next* segment (`i ← i + 1`).

This module defaults to the pseudocode (`causal_step_causal_fail_advance =
True`) and echoes the effective value back in the result dict so downstream
analysis can tell the two regimes apart. The tool does not currently expose a
flag for this; override via the context artifact from Python if you need the
prose regime.

## 14) Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `RuntimeError: CausalStep package not found at …` | `CausalStep/` is not at `imports/CausalStep/`. Restore it there or pass `--causalstep-root /abs/path/to/CausalStep`. |
| `missing artifacts: [...]` in the result dict | One of `causal_step_qa_dataset`, `causal_step_video_id`, `causal_step_answers_cache_dir` is unset. The CLI sets these; if calling `run_modules` directly, populate `EvaluationContext.artifacts`. |
| `no frames read from <path>` | `cv2.VideoCapture` couldn't decode the file. Re-encode with `ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4`. |
| `SegmentDetectionError: no ball impact found ...` | No body whose name contains `"Ball"` had a collision with impulse above `--ball-impulse-threshold`. For controller-driven trajectories, Unity often reports the direct contact with `impulse=0.0`; pass `--ball-impulse-threshold 0.0 --ball-impact-min-time-sec 1.0` (skips startup settling contacts but accepts zero-impulse impact). Otherwise re-record with a stronger push or lower `--ball-impulse-threshold`. |
| `SegmentDetectionError: no fallen domino found ...` | No `Domino_*` body tilted past `--fallen-tilt-deg`. The cascade never propagated — re-record with a stronger push, or lower the threshold. |
| OpenAI 401 / `OPENAI_API_KEY` missing | Export the key in the same shell that runs the CLI. |
| Same prompt, new answers | Cache is keyed by request. If you changed prompt content but kept `causal_step_vlm_prompt_version` the same, delete `<cache-root>/<video_id>/vlm_cache/` and re-run. |
| QA-gen: `--video is required when --mode vlm` | VLM mode reads frames from the Unity GT video. Pass `--video runs/traj_N/unity_gt.mp4`. |
| QA-gen: `RuntimeError: OPENAI_API_KEY must be set` | VLM mode calls OpenAI. Export `OPENAI_API_KEY=sk-...` or rerun without `--mode vlm`. |
| QA-gen: `VLM emitted a distractor identical to the correct answer` | Transient model glitch — `decorate_qa` refused to cache the bad response. Rerun the same command. |
| QA-gen: stale VLM distractors after bumping `--vlm-num-frames` | Cache key includes `num_frames`, so a bump forces a recompute automatically; if you want a clean wipe regardless, delete `<out>.parent/qa_vlm_cache/`. |

## 15) File reference

### WMEval

| File | Purpose |
| --- | --- |
| `evaluation/registry.py` | Module registry; `causal_step` is registered at `MODULE_REGISTRY["causal_step"]`. |
| `evaluation/modules/causal_step.py` | Module entry point: builds the judge, calls the runner for both videos, computes deltas. |
| `evaluation/modules/_causal_step_io.py` | I/O helpers + `ensure_causalstep_importable` (the `sys.path` bootstrap). |
| `evaluation/modules/_causal_step_judge.py` | Generic judge interface (`Judge`, `JudgeRequest`) and frame slicing. |
| `evaluation/modules/_causal_step_openai_judge.py` | Concrete OpenAI VLM judge with on-disk response cache (cache key includes the model id, so swapping `--vlm-model` doesn't collide with prior cached answers). |
| `evaluation/modules/_causal_step_runner.py` | Bridges the judge to CausalStep's `PrecomputedModel` and runs `evaluate_chain` / `evaluate_descriptive_isolated` / `evaluate_causal_isolated`. |
| `evaluation/modules/_causal_step_delta.py` | Polarity-correct delta between Unity and WM metric blocks. |
| `tools/detect_beamdomino_segments.py` | CLI: physics log → `segments.json`. |
| `tools/generate_beamdomino_causal_step_qa.py` | CLI: segments → draft `qa_dataset.json` (templates *or* VLM-hybrid mode; human-review needed). |
| `tools/_causal_qa_vlm.py` | OpenAI VLM-based QA decoration: rewrites question + 3 distractors per segment from the actual frames, and may correct the template's answer when the frames contradict it (overrides logged to stderr). Imported lazily by the QA-gen CLI when `--mode vlm` is set. |
| `tools/evaluate_world_model_causal_step.py` | CLI: manifest → `causal_step_results.json`. |
| `docs/causal_step_qa_review_checklist.md` | What to look for when reviewing the draft QA. |
| `tests/test_causal_step_delta.py` | CausalStep importability + metric polarity. |
| `tests/test_beamdomino_segments.py` | Segment detector unit tests. |
| `tests/test_causal_step_module.py` | Module integration tests with the fake judge. |
| `tests/test_generate_beamdomino_causal_step_qa.py` | QA-gen CLI tests: deterministic shuffle, template/VLM-mode contracts, decorator caching. |

### CausalStep (consumed unchanged)

| File | Purpose |
| --- | --- |
| `imports/CausalStep/causalstep/schema.py` | `VideoInstance`, `QAPair`, `Segment`, `AggregateResults`, `load_dataset`. |
| `imports/CausalStep/causalstep/model_interface.py` | `PrecomputedModel` — what WMEval uses to feed pre-judged answers in. |
| `imports/CausalStep/causalstep/evaluator.py` | `evaluate_chain`, `evaluate_descriptive_isolated`, `evaluate_causal_isolated`. |
| `imports/CausalStep/causalstep/metrics.py` | `compute_all_metrics` — produces the upper-cased metric block. |

## 16) Quick checklist

```
[ ] WMEval/.venv exists; pip install opencv-python numpy openai pytest done.
[ ] OPENAI_API_KEY exported (only for real runs).
[ ] CausalStep/ is at WMEval/imports/CausalStep/.
[ ] pytest tests/test_causal_step_delta.py::test_causalstep_package_importable passes.
[ ] segments.json produced from unity_gt_log.json (each segment > ~0.4s; if any are tiny, re-record or tune detector flags).
[ ] qa_dataset.json produced AND human-reviewed; for multi-run setups, fanned out across N video_ids (one per WM run).
[ ] manifest.jsonl entries carry `trajectory` if you want per-trajectory mean+std in the aggregate.
[ ] Smoke run with --judge fake_always_a passes end-to-end (no API cost).
[ ] Real run with --judge openai (and --wm-full-video if WM is not time-aligned, --vlm-num-frames 8 if any segment is < ~0.5s) produces causal_step_results.json with aggregate + by_trajectory blocks.
```

## 17) Framework compatibility notes

- The module plugs into the shared evaluation context (`reference` +
  `candidate`) like any other WMEval module, but it consumes the videos
  lazily inside the judge rather than through sampled-frame batches — the
  `SampledVideoFrames` objects passed in are stubbed (only `metadata.fps`
  is real; the evaluator probes each video's fps via `cv2.CAP_PROP_FPS`
  before constructing the stub, so Unity 60fps and Genie 20fps clips are
  handled correctly in the same manifest).
- Frames are read from disk inside the judge via `cv2.VideoCapture`
  (`evaluation/modules/causal_step.py::_read_frames_uint8`).
- The CausalStep package itself is consumed unchanged at
  `imports/CausalStep/`; adapter code lives in
  `evaluation/modules/_causal_step_*.py`.

## 18) Scene tuning: weak cascades

If only a handful of dominoes ended up fallen at the end of a recording,
the `cascade` segment will be short and the cascade-stage QA will have
little to discriminate WM from GT — the detector will still produce four
segments, but the `cascade` window may be dominated by post-collapse
stillness.

To get a stronger cascade signal, fix the **scene/physics** side:

- Increase `ballForce` on the BeamDomino agent in Unity (Inspector →
  `BeamDominoAgent`).
- Reduce `dominoSpacing` so adjacent dominoes are within fall reach
  (≤ ~0.6× domino height).
- Visually inspect the recording to confirm the Roof / RoofBall structure
  actually lands on the first domino. If it lands beside the row, no
  cascade can start no matter how strong the push is.

Or tune the **detector** side without re-recording:

- `--fallen-tilt-deg` (default `30°`): lower it to count weakly-tilted
  dominoes as fallen — useful if your scene's dominoes only tip ~15–20°
  before settling against neighbors.
- `--ball-impulse-threshold` (default `0.01`): lower it if your scene
  reports very small impulses on the impact contact.

The detector does not require a particular cascade size to succeed; it
only requires *one* fallen domino to start the topple stage. A trajectory
where 4/60 dominoes fell is a perfectly valid run — just expect lower
discrimination on cascade-segment QA.

## 19) When the world model isn't time-aligned to Unity

The default protocol assumes the WM rollout depicts the same physical
events at roughly the same wall-clock times as the Unity reference.
Project-Genie-style world models that are sampled freely (different runs,
different durations, different camera angles per run, occasional object
duplication) violate that assumption. Two practical adjustments:

1. **Trim the WM clip to a fixed duration** (e.g. 13–15 s, matching
   Unity's recording window) before scoring, so segment-second windows
   don't fall past the end of the clip. If WM clips are also at a
   different fps than Unity, the evaluator handles that automatically
   via per-file `cv2.CAP_PROP_FPS` probing — no extra flag.
2. **Pass `--wm-full-video`** so the WM side judges every QA off the full
   clip rather than against Unity-derived segment windows. Unity side
   keeps its segment windows and the chain/restart machinery is unchanged.
   The trade-off is that the "stepwise" framing is weakened on the WM
   side: the judge can localise the depicted event anywhere in the
   clip, so descriptive-QA accuracy may rise on the WM side relative
   to a strict-windowed protocol. Disclose this in any results that
   compare WM and Unity scores in absolute terms.

**Variance reporting.** When you score N rollouts per trajectory, give
each a distinct `video_id` (`traj_<name>_run<i>`) and add a `trajectory`
field to the manifest entries. The `aggregate.by_trajectory` block then
reports `delta_mean` and `delta_std` per group, which is much more
informative than a single macro-mean across heterogeneous rollouts.

**What this metric does *not* measure.** The QAs probe causal patterns
("ball strikes a leg" / "platform tips" / "dominoes fall in chain"), not
object identity or visual fidelity. A WM that shows wrong objects, extra
debris, or wrong colors but still produces *something* ball-shaped hitting
*something* platform-shaped and *something* tipping over will tend to
match the most-plausible-text answer for each option — and so score close
to Unity on chain-walk metrics. With `--wm-full-video`, this gap widens
because the WM judge sees the entire clip per QA and can cherry-pick a
moment that fits each option. Bumping `--vlm-num-frames` (e.g. to `8`)
narrows the gap: finer sampling within each segment makes the WM's
visual corruption harder to ignore. If you also need a fidelity number,
pair this eval with a separate visual-similarity metric (LPIPS, CLIP-sim)
or add scene-fidelity QAs that explicitly probe object identity / count.
