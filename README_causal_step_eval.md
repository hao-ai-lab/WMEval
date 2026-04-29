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
venv (not in the Unity `mlagents` env):

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

`tools/detect_beamdomino_segments.py` parses the per-frame physics log into the
four canonical BeamDomino stages. The detector derives stage triggers directly
from the recorder's `bodies[].rotation` and `collisions[].impulse` —
`GroundTruthRecorder.cs` does not emit explicit `stack_hit_flag` /
`fallen_count` fields.

| Stage | Boundary rule |
| --- | --- |
| `rolling` | `t=0` until the first frame with a nonzero-impulse collision involving a body whose name contains `"Ball"` (impulse > `--ball-impulse-threshold`, default `0.01`). |
| `impact` | ball impact until the first frame where any `Domino_*` body has tilted past `--fallen-tilt-deg` (default `30°`, computed as `max(|wrap180(rot.x)|, |wrap180(rot.z)|)`). |
| `topple` | first fallen domino until the cascade plateaus (no new fall for `plateau_window_sec`, default `0.5s`). |
| `cascade` | plateau start to end of log. |

```bash
python tools/detect_beamdomino_segments.py \
  --log   runs/traj_001/unity_gt_log.json \
  --out   runs/traj_001/segments.json
```

Optional tuning flags (rarely needed; defaults are tuned to the BeamDomino
scene): `--fallen-tilt-deg`, `--ball-impulse-threshold`, `--plateau-window-sec`.

If the detector raises `SegmentDetectionError`, either no `"Ball"` body ever
collided with sufficient impulse (`no ball impact found ...`) or no domino
ever tilted past the threshold (`no fallen domino found ...`). The
trajectory probably never propagated to a domino fall — re-record with a
stronger ball push, or lower `--ball-impulse-threshold` / `--fallen-tilt-deg`
if the defaults don't match your scene's physics.

## 5) Step 2 — Generate a draft QA dataset (then review it)

`tools/generate_beamdomino_causal_step_qa.py` turns each segment into a
CausalStep `descriptive` + `causal` QA pair using per-stage templates. The
output is a **draft** — the templates are intentionally generic and must be
reviewed by a human.

```bash
python tools/generate_beamdomino_causal_step_qa.py \
  --segments  runs/traj_001/segments.json \
  --video-id  traj_001 \
  --out       runs/traj_001/qa_dataset.json
```

Open `qa_dataset.json` and edit:

- Reword anything too generic to discriminate WM from GT.
- Make sure the `correct_answer` letter still matches after any reorderings.
- Verify segment time spans look sensible against the actual video.

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

The evaluator iterates a JSONL manifest, one trajectory per line. All four keys
are required:

| Key | Meaning |
| --- | --- |
| `video_id` | Stable id used for cache dirs and per-video output rows. |
| `unity_video` | Path to the Unity ground-truth video (reference side). |
| `wm_video` | Path to the world-model candidate video. |
| `qa_dataset` | Path to the CausalStep-schema QA JSON for this trajectory. |

```jsonl
{"video_id":"traj_001","unity_video":"runs/traj_001/unity_gt.mp4","wm_video":"runs/traj_001/wm_pred.mp4","qa_dataset":"runs/traj_001/qa_dataset.json"}
{"video_id":"traj_002","unity_video":"runs/traj_002/unity_gt.mp4","wm_video":"runs/traj_002/wm_pred.mp4","qa_dataset":"runs/traj_002/qa_dataset.json"}
```

Paths can be absolute or relative to the directory you launch the command
from — be consistent across a batch. Per-trajectory failures are caught and
reported as `{"video_id": ..., "error": "..."}` in `per_video` so one bad
trajectory does not halt the batch.

## 8) Step 5 — Dry run with the fake judge on real videos

```bash
python tools/evaluate_world_model_causal_step.py \
  --manifest   runs/manifest.jsonl \
  --cache-root runs/cache \
  --judge      fake_always_a \
  --out        runs/causal_step_results_dry.json
```

This confirms videos load (cv2 reads frames), segments fire, and the delta
block aggregates — all for free.

## 9) Step 6 — Real run with `gpt-4o`

```bash
export OPENAI_API_KEY=sk-...
python tools/evaluate_world_model_causal_step.py \
  --manifest   runs/manifest.jsonl \
  --cache-root runs/cache \
  --judge      openai \
  --out        runs/causal_step_results.json
```

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
      "video_id": "traj_001",
      "unity":  { "CSR": ..., "AMCL": ..., "MCL": ..., "RF": ..., "WS": ..., "DUA": ..., "ICRA": ... },
      "wm":     { "CSR": ..., "AMCL": ..., "MCL": ..., "RF": ..., "WS": ..., "DUA": ..., "ICRA": ... },
      "delta":  { "csr": ..., "amcl": ..., "mcl": ..., "rf": ..., "ws": ..., "dua": ..., "icra": ... },
      "delta_convention": "positive = WM worse than Unity",
      "vlm_model": "gpt-4o",
      "judge_kind": "openai",
      "causal_fail_advance": true
    }
  ],
  "aggregate": {
    "unity_mean":  { "csr": ..., "..." : "..." },
    "wm_mean":     { "csr": ..., "..." : "..." },
    "delta_mean":  { "csr": ..., "..." : "..." },
    "n_trajectories": 2,
    "n_errors": 0,
    "delta_convention": "positive = WM worse than Unity"
  }
}
```

The same `aggregate` block is printed to stdout. `aggregate.delta_mean` is the
headline number per metric.

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
   python tools/detect_beamdomino_segments.py --log .../unity_gt_log.json --out .../segments.json
   python tools/generate_beamdomino_causal_step_qa.py --segments .../segments.json --video-id traj_N --out .../qa_dataset.json
   $EDITOR .../qa_dataset.json   # human review
   echo '{"video_id":"traj_N","unity_video":".../unity_gt.mp4","wm_video":".../wm_pred.mp4","qa_dataset":".../qa_dataset.json"}' >> runs/manifest.jsonl
   python tools/evaluate_world_model_causal_step.py --manifest runs/manifest.jsonl --cache-root runs/cache --judge openai --out runs/causal_step_results.json
   ```
4. Inspect `aggregate.delta_mean`.

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
| `SegmentDetectionError: no ball impact found ...` | No body whose name contains `"Ball"` had a collision with impulse above `--ball-impulse-threshold`. Re-record with a stronger push, or lower the threshold. |
| `SegmentDetectionError: no fallen domino found ...` | No `Domino_*` body tilted past `--fallen-tilt-deg`. The cascade never propagated — re-record with a stronger push, or lower the threshold. |
| OpenAI 401 / `OPENAI_API_KEY` missing | Export the key in the same shell that runs the CLI. |
| Same prompt, new answers | Cache is keyed by request. If you changed prompt content but kept `causal_step_vlm_prompt_version` the same, delete `<cache-root>/<video_id>/vlm_cache/` and re-run. |

## 15) File reference

### WMEval

| File | Purpose |
| --- | --- |
| `evaluation/registry.py` | Module registry; `causal_step` is registered at `MODULE_REGISTRY["causal_step"]`. |
| `evaluation/modules/causal_step.py` | Module entry point: builds the judge, calls the runner for both videos, computes deltas. |
| `evaluation/modules/_causal_step_io.py` | I/O helpers + `ensure_causalstep_importable` (the `sys.path` bootstrap). |
| `evaluation/modules/_causal_step_judge.py` | Generic judge interface (`Judge`, `JudgeRequest`) and frame slicing. |
| `evaluation/modules/_causal_step_openai_judge.py` | Concrete `gpt-4o` judge with on-disk response cache. |
| `evaluation/modules/_causal_step_runner.py` | Bridges the judge to CausalStep's `PrecomputedModel` and runs `evaluate_chain` / `evaluate_descriptive_isolated` / `evaluate_causal_isolated`. |
| `evaluation/modules/_causal_step_delta.py` | Polarity-correct delta between Unity and WM metric blocks. |
| `tools/detect_beamdomino_segments.py` | CLI: physics log → `segments.json`. |
| `tools/generate_beamdomino_causal_step_qa.py` | CLI: segments → draft `qa_dataset.json` (human-review needed). |
| `tools/evaluate_world_model_causal_step.py` | CLI: manifest → `causal_step_results.json`. |
| `docs/causal_step_qa_review_checklist.md` | What to look for when reviewing the draft QA. |
| `tests/test_causal_step_delta.py` | CausalStep importability + metric polarity. |
| `tests/test_beamdomino_segments.py` | Segment detector unit tests. |
| `tests/test_causal_step_module.py` | Module integration tests with the fake judge. |

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
[ ] segments.json produced from unity_gt_log.json.
[ ] qa_dataset.json produced AND human-reviewed.
[ ] Smoke run with --judge fake_always_a passes end-to-end.
[ ] Real run with --judge openai produces causal_step_results.json with aggregate block.
```

## 17) Framework compatibility notes

- The module plugs into the shared evaluation context (`reference` +
  `candidate`) like any other WMEval module, but it consumes the videos
  lazily inside the judge rather than through sampled-frame batches — the
  `SampledVideoFrames` objects passed in are stubbed.
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
