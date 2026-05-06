# LiquidContainer Evaluation README

This document explains how to run the LiquidContainer (sealed-bottle
sloshing) evaluation in the current `WMEval` framework end-to-end: from a
Unity reference recording and a world-model candidate video through SAM2
liquid/bottle masking, RAFT velocity fields, the four sloshing modules
(plus the cross-cutting perceptual modules), and finally the per-model
JSON report and pooled leaderboard CSV.

The module set treats the bottle as a **sealed half-filled container**
under fixed-camera (regime `c2_static_side`) playback, so failures are
*continuous* — volume drift, surface-tilt residual, off-frequency slosh —
rather than discrete event errors. Every metric is reported as a `{unity,
wm, delta}` block with the convention **positive delta = WM is worse than
Unity** (`delta = wm − unity`).

## 1) Pipeline overview

The pipeline is a stack of shell wrappers around one Python entry point.
All four sloshing modules share the same SAM2 + RAFT cache, so a
manifest run pays mask/flow cost only once per video:

```
scripts/normalize_fps.sh                  (sourced helper, not run alone)
    in:  REF_VIDEO, WM_VIDEO              (any resolution / fps)
    out: 832x464 @ 16 fps mp4s            (WMEval LiquidContainer canonical)

scripts/evaluate_world_model_liquid_full.sh
    in:  REF_VIDEO, WM_VIDEO, OUT_JSON [, WM_NAME]
    out: per-pair report.json             (4 sloshing modules + lpips + control_motion + clip4clip)

scripts/run_liquid_leaderboard.sh
    in:  MANIFEST_TSV, OUT_DIR [, REF_VIDEO_DEFAULT]
    out: <OUT_DIR>/<model>.json (one per row) + leaderboard.csv

scripts/sanity_unity_vs_unity_liquid.sh
    in:  REF_VIDEO [CAND_VIDEO] [OUT]
    out: identity.json                    (the noise floor; read deltas RELATIVE to this)
```

The Python entry point under those wrappers is
[tools/evaluate_world_model_liquid.py](tools/evaluate_world_model_liquid.py),
which thinly wraps `evaluation.runner.run_world_model_evaluation` with
the `--liquid_*` argument namespace. The Unity recording side (running
the LiquidContainer scene to produce `Unity_<profile>.mp4`) lives in
`WorldModelBenchScenePrivate/GameScenes/LiquidContainer/`. This README
is scoped to everything that happens once you have a Unity reference
mp4 and a world-model candidate mp4 on disk.

## 2) Scope

- Main entry: [tools/evaluate_world_model_liquid.py](tools/evaluate_world_model_liquid.py)
- Wrappers:
  - [scripts/evaluate_world_model_liquid_full.sh](scripts/evaluate_world_model_liquid_full.sh) — single (REF, WM) pair, full headline set.
  - [scripts/run_liquid_leaderboard.sh](scripts/run_liquid_leaderboard.sh) — fleet of WMs from a TSV manifest → CSV.
  - [scripts/sanity_unity_vs_unity_liquid.sh](scripts/sanity_unity_vs_unity_liquid.sh) — Unity-vs-Unity noise floor.
- Mask debugging: [tools/liquid_mask_qa.py](tools/liquid_mask_qa.py) — render SAM2 liquid + bottle overlays at first / mid / last frame to verify the prompt before paying for a full eval.
- Report inspection: [tools/inspect_liquid_report.py](tools/inspect_liquid_report.py) — pretty-print one or many reports with sanity flags.
- The four sloshing modules registered in [evaluation/registry.py](evaluation/registry.py):
  - `liquid_velocity` — RAFT (default) or OpenPIV flow inside the liquid mask.
  - `liquid_volume_conservation` — projected mask area jitter + drift.
  - `liquid_surface_orientation` — air/liquid interface tilt RMS.
  - `slosh_spectrum` — meniscus FFT, dominant frequency + spectral L1.
- Cross-cutting (used by the leaderboard wrapper, optional otherwise):
  `lpips`, `control_motion`, `clip4clip`.

## 3) Environment

The four sloshing modules require:

- **SAM2** (`facebook/sam2.1-hiera-small` by default), pulled lazily by
  [evaluation/modules/_liquid_common.py](evaluation/modules/_liquid_common.py)
  on first use; needs HF cache + GPU.
- **GroundingDINO** to seed SAM2's first-frame box from the text prompt
  (`"the liquid inside the bottle"` / `"the glass bottle"`).
- **RAFT** (default) or **OpenPIV** for in-mask velocity estimation.
- **ffmpeg / ffprobe** on `PATH` for `normalize_fps`.
- `opencv-python`, `numpy`, `scipy`, `torch` in the WMEval venv.

Create / activate the WMEval venv (do **not** reuse the Unity `mlagents`
env — its pinned `numpy` / `protobuf` versions fight `opencv-python` /
`torch`):

```bash
cd WMEval_code/WMEval
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # if present, otherwise install the imports above
```

If you plan to enable `lpips` (it is on in the full wrapper), the VLM
judge instructions are scoped via `--lpips_vlm_scene_key liquid_container`
so the prompt knows it's looking at a sloshing bottle.

## 4) Canonical resolution / fps

LiquidContainer canonical is **832×464 @ 16 fps** — lingbot's native
output (the constrained side). Unity (1280×720 @ 30 fps) and any other
WM recording are downsampled to match before the eval runs.

[scripts/normalize_fps.sh](scripts/normalize_fps.sh) provides
`normalize_video` (used by both `evaluate_world_model_liquid_full.sh` and
`sanity_unity_vs_unity_liquid.sh`). Override via env vars:

```bash
WMEVAL_CANONICAL_FPS=16 WMEVAL_CANONICAL_W=832 WMEVAL_CANONICAL_H=464 \
  bash scripts/evaluate_world_model_liquid_full.sh ref.mp4 wm.mp4 out.json
```

Without normalization:
- PIV / RAFT grid shapes differ across WMs, so EPE / cosine break.
- An fps mismatch silently scales pixel velocities and FFT frequencies
  by `ref_fps / wm_fps`. `_liquid_common.require_matched_fps` raises
  rather than letting that scale through, but only when both sides
  reach the modules — normalize first to avoid the late failure.

## 5) Step 1 — Mask QA (cheap, do this first on any new video set)

`liquid_mask_qa.py` runs SAM2 with both the liquid prompt and the bottle
prompt and writes overlay PNGs at first / middle / last frame. A wrong
mask poisons all four sloshing metrics silently — always eyeball this on
at least one trajectory per WM before paying for a full leaderboard.

```bash
python tools/liquid_mask_qa.py \
  --videos /path/to/wm_pred.mp4 \
  --out reports/liquid_qa/wm_pred \
  --liquid-prompt "the liquid inside the bottle" \
  --bottle-prompt "the glass bottle"
```

Inspect `reports/liquid_qa/wm_pred/summary_strip.png`. The cyan overlay
should hug the liquid only (not the glass walls); the blue overlay
should match the full bottle silhouette. If either is wrong, reword the
prompt and rerun (cache is keyed on prompt content).

## 6) Step 2 — Noise-floor sanity (Unity vs Unity)

Before reading any WM number, run the floor finder. The deltas it
produces are the irreducible Unity-vs-Unity noise (render seed,
stochastic shader state, RAFT non-determinism); every WM-vs-Unity number
should be read **relative to this**, not against zero.

```bash
# self-pair: same clip on both sides; deltas should be ~0.
bash scripts/sanity_unity_vs_unity_liquid.sh

# cross-pair: two seeds of the same Unity scene.
bash scripts/sanity_unity_vs_unity_liquid.sh \
  unity/LiquidContainer/Recordings/Movie_001.mp4 \
  unity/LiquidContainer/Recordings/Movie_002.mp4 \
  reports/world_model_liquid_sanity/cross_pair.json
```

The expected ranges for each metric on a self-pair vs cross-pair are
documented in the script header. If self-pair deltas are not near zero,
something upstream is broken — fix that before trusting WM deltas.

## 7) Step 3 — Single-pair eval

The "full" wrapper runs all four sloshing modules plus the cross-cutting
perceptual modules and tags the report with the model name:

```bash
bash scripts/evaluate_world_model_liquid_full.sh \
  unity/LiquidContainer/Recordings/Unity_step_hold.mp4 \
  outputs/lingbot/.../step_hold-.../video.mp4 \
  reports/liquid/lingbot_step_hold.json \
  lingbot_run1_step_hold
```

Equivalent direct invocation (skips fps normalization — be sure both
videos are already at canonical 832×464 @ 16 fps):

```bash
python tools/evaluate_world_model_liquid.py \
  --video         outputs/.../video_norm.mp4 \
  --ref_video     unity/.../Unity_step_hold_norm.mp4 \
  --modules       liquid_velocity,liquid_volume_conservation,liquid_surface_orientation,slosh_spectrum \
  --liquid_cache_root  runs/liquid_cache \
  --liquid_sam2_model  facebook/sam2.1-hiera-small \
  --liquid_sam2_prompt "the liquid inside the bottle" \
  --liquid_bottle_prompt "the glass bottle" \
  --liquid_flow_estimator raft \
  --liquid_camera_compensation \
  --liquid_camera_regime c2_static_side \
  --output_json   reports/liquid/lingbot_step_hold.json \
  --eval_fps 4 --max_pairs 512 --liquid_max_pairs 0
```

Inspect:

```bash
python tools/inspect_liquid_report.py reports/liquid/lingbot_step_hold.json
```

## 8) Step 4 — Build a manifest

The leaderboard wrapper iterates a TAB-separated manifest, one WM per
line. Lines starting with `#` and blank lines are skipped. CRLF-encoded
manifests are tolerated.

| Column | Meaning |
| --- | --- |
| 1 | `model_name` — used as filename for the per-model JSON, must be unique. |
| 2 | `candidate_video_path` — the WM mp4. |
| 3 (optional) | `ref_video_override` — Unity reference for THIS row. Required when different rows pair against different references (e.g. one ref per action profile). |

Example (excerpt of [manifests/liquid_full_grid.tsv](manifests/liquid_full_grid.tsv)):

```tsv
lingbot_run1_step_hold	/path/to/wm_step_hold.mp4	/path/to/Unity_step_hold.mp4
matrix_game2_run1_rest	/path/to/wm_rest.mp4	/path/to/Unity_rest.mp4
```

Pre-built manifests:
- [manifests/liquid_full_grid.tsv](manifests/liquid_full_grid.tsv) — 64 rows: lingbot run1-6 + matrix-game2 run1-5 + matrix-game3 run1-5 × 4 action profiles.
- [manifests/lingbot_liquid_grid.tsv](manifests/lingbot_liquid_grid.tsv) — lingbot subset.
- [manifests/matrix_games_liquid_grid.tsv](manifests/matrix_games_liquid_grid.tsv) — matrix-game subset.

## 9) Step 5 — Leaderboard run

```bash
bash scripts/run_liquid_leaderboard.sh \
  manifests/liquid_full_grid.tsv \
  reports/world_model_liquid_full
```

Outputs:

```
reports/world_model_liquid_full/
  <model>.json          # one per row (full per-module report; tagged with "model")
  leaderboard.csv       # five-pillar headline table, one row per model
```

Per-row failures are caught: the row is logged to stderr and the rest of
the manifest still runs. The summary at the end lists failed rows.

The CSV columns (driven by the wrapper's `COLUMNS` list) are
deliberately the **headline** set:

```
model,
liquid_velocity.cosine_similarity,
liquid_volume_conservation.mask_area_std_normalized.delta,
liquid_volume_conservation.mask_area_drift_per_sec.delta,
liquid_surface_orientation.surface_tilt_residual_deg.delta,
slosh_spectrum.meniscus.spectral_l1,
slosh_spectrum.meniscus.dominant_frequency_hz.delta_hz,
lpips.lpips_mean
```

EPE / divergence_residual / curl are intentionally absent from the CSV:
on opaque uniform liquids RAFT magnitudes are dominated by hallucinated
flow, so those numbers track estimator noise rather than physics. They
remain in each per-model JSON for identity-pair sanity checks.

## 10) Caches

All SAM2 masks (liquid + bottle), bottle centroid traces, and PIV / RAFT
fields are pickled under:

```
runs/liquid_cache/
  <video_basename>__<content_hash>__<kind>__<config_hash>.pkl
```

The cache key includes the video's content hash and a per-kind config
hash (only the keys that affect each output are hashed — changing PIV
window size does not invalidate SAM2 masks). To force a recompute, delete
the matching pickle. Sharing the same `--liquid_cache_root` across
trajectories is intentional: each video is masked once across all four
modules and across re-runs.

## 11) Output fields and interpretation

A per-pair report.json has the shape:

```json
{
  "model": "lingbot_run1_step_hold",
  "results": {
    "liquid_velocity": {
      "epe": { "mean": ..., "median": ..., "p90": ... },
      "cosine_similarity": ...,
      "divergence_residual": ...,
      "curl_stats": { "mean_error": ..., "std_error": ... },
      "temporal_consistency": ...,
      "diagnostics": {
        "flow_estimator": "raft",
        "piv_validity_unity": ..., "piv_validity_wm": ...,
        "liquid_mask_unity": { "mean_area_frac": ... },
        "liquid_mask_wm":    { "mean_area_frac": ... },
        "bottle_compensation": true
      }
    },
    "liquid_volume_conservation": {
      "mask_area_std_normalized": { "unity": ..., "wm": ..., "delta": ... },
      "mask_area_drift_per_sec":  { "unity": ..., "wm": ..., "delta": ... },
      "diagnostics": { "frames_evaluated": N }
    },
    "liquid_surface_orientation": {
      "surface_tilt_residual_deg": { "unity": ..., "wm": ..., "delta": ... }
    },
    "slosh_spectrum": {
      "meniscus":  { "dominant_frequency_hz": { "unity": ..., "wm": ..., "ratio": ..., "delta_hz": ... },
                     "spectral_l1": ... },
      "centroid":  { ... same shape ... }
    },
    "lpips": { "lpips_mean": ..., ... },
    "control_motion": { ... },
    "clip4clip":      { ... }
  }
}
```

### Metric reference

For every `delta` field below, **positive = WM worse than Unity**.

- `liquid_velocity.cosine_similarity` — best at `1` (worst at `-1`).
  EPE / divergence_residual / curl_stats / temporal_consistency are kept
  in the JSON for identity-pair sanity but are **not** the headline
  numbers — see §9.
- `liquid_volume_conservation.mask_area_std_normalized.delta` — short-
  horizon mask flicker; ideal `0`.
- `liquid_volume_conservation.mask_area_drift_per_sec.delta` — long-
  horizon leakage / spurious growth (1/sec); ideal `0`. A sealed bottle
  with a fixed camera should have area exactly conserved up to AA noise.
- `liquid_surface_orientation.surface_tilt_residual_deg.delta` — extra
  RMS tilt (degrees) of the air/liquid interface beyond Unity's. For an
  upright bottle Unity is ~0; a WM that tilts the surface gets penalised.
- `slosh_spectrum.meniscus.spectral_l1` — L1 distance between unit-area
  power spectra (catches mode-shape mismatch even when the peak frequency
  is right). Ideal `0`.
- `slosh_spectrum.meniscus.dominant_frequency_hz.delta_hz` — peak-
  frequency offset in Hz. Ideal `0`.

The `meniscus` block is preferred over `centroid` because the centroid
is contaminated by camera-induced bottle motion already absorbed by
volume / orientation.

## 12) Typical iteration loop

1. Record one Unity reference per action profile from
   `WorldModelBenchScenePrivate/GameScenes/LiquidContainer/` →
   `unity/LiquidContainer/Recordings/Unity_<profile>.mp4`.
2. Generate WM rollouts conditioned on the same first frame / actions →
   `outputs/<wm>/LiquidContainer/<run>/<profile>-<ts>/video.mp4`.
3. Eyeball masks for one trajectory per WM:
   ```bash
   python tools/liquid_mask_qa.py --videos outputs/.../video.mp4 --out reports/liquid_qa/<wm>
   ```
4. Build / extend the manifest:
   ```bash
   echo -e "<wm>_<run>_<profile>\t<wm_path>\t<unity_path>" >> manifests/liquid_full_grid.tsv
   ```
5. Run the leaderboard:
   ```bash
   bash scripts/run_liquid_leaderboard.sh manifests/liquid_full_grid.tsv reports/world_model_liquid_full
   ```
6. Inspect:
   ```bash
   python tools/inspect_liquid_report.py reports/world_model_liquid_full/<wm>.json
   column -t -s, reports/world_model_liquid_full/leaderboard.csv | less -S
   ```

## 13) Camera regimes

The shipped wrappers assume `c2_static_side` — fixed camera, bottle
stationary in world. Camera bulk motion is then negligible, and
`--liquid_camera_compensation` (on by default) subtracts the SAM2 bottle
centroid delta from the velocity field to absorb residual rendering
jitter. For other regimes (e.g. `c1_handheld`):

- Bottle centroid will drift across the frame — bulk-motion compensation
  becomes load-bearing instead of cosmetic. Confirm bottle masks are
  clean before trusting velocity numbers.
- The slosh-spectrum centroid trace should not be used (camera motion
  dominates); the `meniscus` trace is camera-compensated by tracking the
  bottle's own y-coordinate. Stick to the `meniscus` headline number.
- `liquid_volume_conservation` still works but interpret short-horizon
  jitter relative to the camera-only sanity floor for that regime.

Pass `--liquid_camera_regime c1_handheld` to tag the report; nothing else
in the modules currently branches on it (the field is informational), but
downstream analysis can filter on it.

## 14) RAFT vs OpenPIV

`--liquid_flow_estimator raft` (default) uses learned dense optical flow;
works on opaque / uniform fluids where the texture-free interior would
otherwise return NaN under cross-correlation. `--liquid_flow_estimator
openpiv` uses cross-correlation; only meaningful when the liquid has rich
internal texture (e.g. milky / particulate fluids). On the WMEval
LiquidContainer scene the default is RAFT.

If you switch estimators, also drop the cached PIV pickles for the
affected videos — the cache is keyed on (window_size, overlap), not on
estimator identity.

## 15) Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `ValueError: liquid_*: reference fps != candidate fps` | One side wasn't run through `normalize_fps`. Use the wrappers in `scripts/`, or set `WMEVAL_CANONICAL_FPS` consistently. |
| `flow validity wm = X% vs mask coverage Y% (ratio < 0.5)` from `inspect_liquid_report.py` | RAFT/PIV is rejecting cells INSIDE the mask. With `openpiv`: try `--liquid_flow_estimator raft`. With `raft`: re-run mask QA — the mask probably tracks the wrong region (glass walls, label, reflection). |
| All deltas non-zero on a Unity-vs-Unity self-pair | Either the wrapper is reusing a stale cache from a prior incompatible config, or non-determinism is leaking in. Delete `runs/liquid_cache/` for that video and rerun `sanity_unity_vs_unity_liquid.sh`. |
| `slosh_spectrum.meniscus.dominant_frequency_hz` near `eval_fps/2` | Aliased. Either the cavity's natural mode is above Nyquist at the canonical fps, or the meniscus trace is dominated by per-frame mask jitter — check mask QA. |
| `liquid_volume_conservation.mask_area_drift_per_sec` huge but mask QA looks right | SAM2 may be losing the liquid mid-clip. Inspect `summary_strip.png` from `liquid_mask_qa.py` at 1st / mid / last frame; if the last-frame mask is wrong, raise the prompt specificity or re-record the WM clip. |
| GroundingDINO finds no liquid on first frame | The WM may have hallucinated a glitched first frame. Skip the row in the manifest — `run_liquid_leaderboard.sh` already swallows this and continues. |
| Leaderboard CSV has empty cells in a column | Module errored on that row. Open the per-model JSON to see the `error` key under that module's block. |
| `cv2.VideoCapture` returns 0 frames | Re-encode: `ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4`. |

## 16) File reference

### WMEval

| File | Purpose |
| --- | --- |
| [evaluation/registry.py](evaluation/registry.py) | Module registry; `liquid_velocity`, `liquid_volume_conservation`, `liquid_surface_orientation`, `slosh_spectrum` are all registered here. |
| [evaluation/modules/_liquid_common.py](evaluation/modules/_liquid_common.py) | Shared schema (`LiquidMaskResult`, `PIVField`, `BottleTrace`), SAM2 / GroundingDINO / RAFT / OpenPIV cached I/O, fps-match guard, divergence/curl helpers. |
| [evaluation/modules/liquid_velocity.py](evaluation/modules/liquid_velocity.py) | Velocity field metrics: EPE, cosine, divergence/curl residuals, temporal consistency. |
| [evaluation/modules/liquid_volume_conservation.py](evaluation/modules/liquid_volume_conservation.py) | Mask-area std + drift, with `{unity, wm, delta}` per stat. |
| [evaluation/modules/liquid_surface_orientation.py](evaluation/modules/liquid_surface_orientation.py) | Per-frame topmost-pixel-per-column line fit → tilt RMS. |
| [evaluation/modules/slosh_spectrum.py](evaluation/modules/slosh_spectrum.py) | Centroid + meniscus FFT; dominant-frequency / spectral L1. |
| [tools/evaluate_world_model_liquid.py](tools/evaluate_world_model_liquid.py) | CLI: single-pair eval, `--liquid_*` namespace. |
| [tools/liquid_mask_qa.py](tools/liquid_mask_qa.py) | CLI: SAM2 mask overlay PNGs at 1st / mid / last frame. |
| [tools/inspect_liquid_report.py](tools/inspect_liquid_report.py) | CLI: pretty-print one or many reports with sanity flags. |
| [scripts/normalize_fps.sh](scripts/normalize_fps.sh) | Sourced helper: `normalize_video` to 832×464 @ 16 fps. |
| [scripts/evaluate_world_model_liquid_full.sh](scripts/evaluate_world_model_liquid_full.sh) | Single-pair wrapper: normalize + eval + tag. |
| [scripts/run_liquid_leaderboard.sh](scripts/run_liquid_leaderboard.sh) | Manifest fan-out → per-model JSON + `leaderboard.csv`. |
| [scripts/sanity_unity_vs_unity_liquid.sh](scripts/sanity_unity_vs_unity_liquid.sh) | Unity-vs-Unity noise floor. |
| [manifests/liquid_full_grid.tsv](manifests/liquid_full_grid.tsv) | 64-row paper grid (lingbot + matrix-game2 + matrix-game3). |
| [tests/test_liquid_registry.py](tests/test_liquid_registry.py) | Module registration smoke. |
| [tests/test_liquid_common_cache.py](tests/test_liquid_common_cache.py) | Cache-key invariants. |
| [tests/test_liquid_common_divcurl.py](tests/test_liquid_common_divcurl.py) | Divergence / curl pure-numerics. |
| [tests/test_liquid_velocity_metrics.py](tests/test_liquid_velocity_metrics.py) | EPE / cosine / div-curl / temporal-consistency stats. |
| [tests/test_liquid_volume_conservation.py](tests/test_liquid_volume_conservation.py) | Mask-area std / drift. |
| [tests/test_liquid_surface_orientation.py](tests/test_liquid_surface_orientation.py) | Per-frame tilt + RMS. |

### Unity scene (consumed unchanged)

| File | Purpose |
| --- | --- |
| `WorldModelBenchScenePrivate/GameScenes/LiquidContainer/` | Unity project: action profiles, sealed bottle scene, Recordings/. |
| `Recordings/Unity_<profile>.mp4` | Per-profile Unity reference (one per action profile). |

## 17) Quick checklist

```
[ ] WMEval/.venv exists; opencv, numpy, torch, transformers installed.
[ ] ffmpeg + ffprobe on PATH (normalize_fps requires both).
[ ] HF cache reachable and big enough for facebook/sam2.1-hiera-small.
[ ] Mask QA run on at least one trajectory per WM; overlays look right.
[ ] sanity_unity_vs_unity_liquid.sh self-pair returns deltas ≈ 0.
[ ] sanity_unity_vs_unity_liquid.sh cross-pair logged as the noise floor.
[ ] manifest TSV contains TAB separators, one row per (model_name, wm_video, ref_video).
[ ] run_liquid_leaderboard.sh produces leaderboard.csv with no FAILED rows.
[ ] inspect_liquid_report.py shows no flagged validity / NaN issues on the headline rows.
```

## 18) Framework compatibility notes

- The four sloshing modules plug into the shared evaluation context
  (`reference` + `candidate`) like any other WMEval module, but they
  consume the videos lazily inside SAM2 / RAFT rather than through the
  sampled-frame batches the LPIPS / clip4clip path uses. The
  `SampledVideoFrames` objects in the context are still required (their
  `metadata.fps` is what the fps-match guard reads).
- All four modules read from the **same** SAM2 mask cache for a given
  video; adding a new mask-driven module is cheap.
- `liquid_velocity` is the only module with a hard estimator dependency
  (RAFT or OpenPIV). The other three only need the SAM2 mask, so they
  work even when the velocity module is dropped from `--modules`.
