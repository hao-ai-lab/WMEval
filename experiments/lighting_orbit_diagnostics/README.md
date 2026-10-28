# Lighting orbit diagnostics

> Historical (2026-10-10): the trials that led to the retired explicit-orbit
> protocol in `experiments/lighting_orbit/`. Lite lighting is the paper's native
> case `SunRisesAndSets/lighting` (2026-10-20); see docs/evaluation_dimensions.md.

Completed diagnostic trials, 2026-10-10; release integration remains open.
These trials test the proposed release action; they do not replace the existing
paper videos or the locked lighting case. Only LingBot and Waypoint have been
tested with the new controls, each at seed 42.

The primary action is a 10-second, 360-degree CCW camera orbit at radius 2 m,
height 1.6 m, looking at the table centre at height 0.75 m, with fixed sunlight.
`tools/trajectories/prepare_lighting_orbit.py` exports OpenCV camera-to-world poses, camera
intrinsics, timestamped JSONL, hashes and a preview derived from those arrays.
The geometry is explicit; there is no implicit translation hidden in a yaw
channel. Scene scale, target height and FOV are nominal assumptions pending
image/scene calibration. The seed frame remains the existing case's `ref.png`.

## Trials

- `run_lighting_orbit_modal.py`: LingBot Base Cam, one seed, 161 explicit poses,
  20 sampling steps, eight H100s, 30-minute function timeout. The source revision
  is pinned; cached weights have an unrecorded HF revision and this limitation
  is captured in the receipt. No action-string-to-pose substitution is used.
- `run_lighting_waypoint_probe_modal.py`: one H100, paired seed-42 probes for
  yaw alone, D-strafe alone and their combination, 45 control steps each.
  Waypoint emits four frames per step at 60 fps, plus four conditioning frames.
  Keyboard/mouse magnitudes are diagnostic, not a claim of metres or degrees.

Results are stored under `WMEval-lite-candidates:lighting_orbit_diagnostics/`
in separate run directories, alongside commands, controls, hashes and logs.
`tools/analysis/review_lighting_video.py` makes timestamped contact sheets from downloaded
videos. Inspect both the scene and camera motion before promoting a control
program to a benchmark action. Failure to follow a supported pose command is a
model result; it is not automatically proof that the interface lacks support.

| Run directory / video | Actual video | Observation from decoded contact sheets |
| --- | --- | --- |
| `lingbot_orbit360_20261010_seed42_v2/video.mp4` | 161 frames, 16 fps, 624×624, 10.0625 s | Clear orbital viewpoint changes; table largely stays central. Final view does not return to the initial view; objects change and a bright disc appears in the sky. The commanded path closes, but realized 360-degree closure is not established. |
| `waypoint_probes_20261010_seed42/yaw_only.mp4` | 184 frames, 60 fps, 1280×720, 3.0667 s | Tabletop remains visible during the short turn. Candidate fallback: 45 steps, no D key, mouse x = −0.6. This is a short yaw-only action, not an orbit. |
| Same run, `strafe_only.mp4` and `strafe_yaw.mp4` | Same frame metadata | Sustained D translation loses the table quickly; adding mouse x = −0.6 does not keep it centred. |
| `waypoint_workarounds_20261010_seed42/combined_duty.mp4` | 604 frames, 60 fps, 10.0667 s | D key at 10% pulse duty plus constant yaw reduces translation, but the table drifts right and clips near the end. |
| Same run, `yaw_sweep.mp4` | Same frame metadata | No D; cosine mouse sweep, peak magnitude 0.6. Table clips in the middle despite nominal zero net mouse input. |
| `waypoint_yaw_sweep_small_20261010_seed42/yaw_sweep.mp4` | Same frame metadata | Peak magnitude reduced to 0.3; table still clips in the middle. Halving input magnitude does not establish a proportionate reduction in realized angle. |

The short Waypoint yaw is the current fallback candidate. It has a different
duration and motion condition, and must have its own baseline; its scores
cannot be pooled with the 10-second orbit. No realized angle or formal scene
retention score has been estimated for these diagnostic videos. The visual
observations above are not six-model or five-seed benchmark results. LingBot
used 20 sampling steps; its result does not prove failure under other settings.
Waypoint used the paper's square-to-16:9 stretched seed input and no text prompt.
Neither trial records a complete weight revision/hash manifest.

The pinned [Waypoint source](https://github.com/Overworldai/world_engine/blob/b3f1e725b222679a517632918cc78bba0c9fa433/src/model/world_model.py)
passes continuous mouse values into the control embedding. The trials provide
no basis for asserting that this input has a calibrated angle gain.

`results.json` records the downloaded video checksums, frame metadata, controls
checksums, Modal application IDs and nominal program parameters. Each cloud run
also retains its receipt; the LingBot run retains the generation log.

If a full orbit cannot be expressed or is unsuitable for the shared protocol,
test a smaller look-at arc first, then a bounded yaw-only sweep if necessary.
Any fallback needs a distinct protocol label and its own baseline; different
motion conditions must not be silently mixed into a matched-action comparison.

## Reproduce the diagnostics

These commands require the existing Modal profile, `wm-hf-cache`, the
`WMEval-lite-candidates` volume and the `huggingface` secret for Waypoint.
Use a fresh run ID: the runners refuse to overwrite an existing cloud run.

```sh
python tools/trajectories/prepare_lighting_orbit.py --output /tmp/wmeval-orbit-input --preview
WMEVAL_ORBIT_INPUT=/tmp/wmeval-orbit-input modal run tools/generation/run_lighting_orbit_modal.py --run-id YOUR_FRESH_ORBIT_RUN
modal run tools/generation/run_lighting_waypoint_probe_modal.py --mode probes --steps 45 --run-id YOUR_FRESH_PROBE_RUN
modal run tools/generation/run_lighting_waypoint_probe_modal.py --mode compare_fallbacks --steps 150 --mouse-x -0.6 --strafe-duty 0.1 --run-id YOUR_FRESH_COMPARISON_RUN
modal run tools/generation/run_lighting_waypoint_probe_modal.py --mode yaw_sweep --steps 150 --mouse-x -0.3 --run-id YOUR_FRESH_SMALL_SWEEP_RUN
```

To pin the historical native baseline again, pass the downloaded score JSONL
from `lite-lighting-release-cleanup-20261009`:

```sh
modal run tools/analysis/pin_lighting_paper_baseline_modal.py --output /tmp/lighting_paper_native.json --scores /absolute/path/per_case_results.jsonl
```

This CPU-only job verifies all 30 stored videos against the source volume's
manifest before attaching the six native metrics by video identity. Genie
records keep `seed: null` and a separate hosted `run_index`.

## Frame rate

`candidate_normalization: lighting_fps12` is available in the Lite runner.
It uses timestamps to produce 12 fps before detection/tracking, preserves frame
size and playback duration (to frame-grid precision), and uses lossless H.264.
The existing case remains `none` until the new protocol and baseline are ready.

The previous scratch analysis used integer strides: its reported '12 fps'
results actually compared Genie at 10 fps, LingBot at 16, MG3 at 17 and HY/
Waypoint at 12. Those results cannot establish the common-fps ranking.

## Checks already completed

- Orbit closure, constant radius/height, table on the optical axis, proper
  rotation determinant, and square-seed intrinsic conversion.
- Real ffmpeg normalization tests for inputs at 12/16/17/20/24/60 fps.
- 107 relevant tests pass; the original suite's 32 cases validate.
- Original saved September/October runs: 30 rows completed, all 180 metric
  values identical; 25 saved tracks recomputed with current code match all
  150 metric values. This verifies rescoring existing videos, not regeneration
  under the new action.
- All 30 original video hashes match the volume's manifest. Native frame
  metadata and metrics are pinned in
  `lite/baselines/lighting_paper_native.json`. This pins the
  Modal source; it does not constitute publication of the files to HF.
