# Shadow Consistency Evaluation

Evaluate lighting and shadow consistency of world-model-generated videos using
GroundingDINO + SAM 2 object tracking and Hu-moment-based shadow-shape
analysis. No per-frame Unity ground-truth reference is required.

## Quick Start

```bash
python tools/evaluate_world_model_shadow_consistency.py \
    --video path/to/generated_video.mp4 \
    --text_prompt "table. chair. lamp." \
    --anchor_label table \
    --hud_preset genie \
    --output_json reports/shadow_consistency.json
```

Or via the unified runner:

```bash
python tools/evaluate_world_model_diffusionlight.py \
    --video path/to/generated.mp4 \
    --ref_video path/to/unity_ref.mp4 \
    --modules shadow_consistency \
    --output_json reports/shadow_consistency.json
```

## Metrics

| Metric | Abbrev | Direction | Description |
|--------|--------|-----------|-------------|
| Scene Retention Rate | SRR | ↑ | Fraction of frames where the anchor object remains on screen |
| Shadow Detection Rate | SDR | ↑ | Of scene-retained frames, fraction with ≥1 valid shadow |
| Multi-Object Sun Disagreement | MOSD | ↓ | Per-frame circular std of per-object shadow azimuths (degrees) |
| Shadow Direction Jitter | SDJ | ↓ | Median absolute frame-to-frame step in mean shadow azimuth (deg/frame) |
| Shadow Area Stability | SAS | ↓ | Mean relative shadow-area change between consecutive frames (%) |
| Shadow Shape Drift | SSD | ↓ | Mean L2 distance between consecutive log-Hu-moment vectors |

## Pipeline

1. **Object detection** (frame 0): GroundingDINO (`IDEA-Research/grounding-dino-tiny`)
   detects anchor objects from a typed text prompt.
2. **Video tracking**: SAM 2 (`facebook/sam2.1-hiera-large`) propagates detected
   object masks through all frames.
3. **Shadow extraction**: For each tracked object, a luminance gate inside a
   dilated halo extracts the cast shadow on the floor. Connected-component
   filtering picks the shadow blob closest to the object.
4. **Hu moments**: Translation/rotation/scale-invariant shape descriptors on the
   shadow mask. A smooth orbital perspective change does not register; only
   genuine shape morphing does.
5. **Metric computation**: Six metrics from the per-frame track (see above).

## HUD Presets

Model-specific HUD overlays (keyboard hints, watermarks) are masked out before
detection to avoid polluting the pipeline:

| Preset | Model | What it masks |
|--------|-------|---------------|
| `none` | (default) | Nothing |
| `lingbot` | LingBot-World | WASD (bottom-left) + IJKL (bottom-right) |
| `mg2` | Matrix Game 2.0 | Bottom-row icons |
| `mg3` | Matrix Game 3.0 | Cursor + bottom-centre keys |
| `hy` | HY-WorldPlay | Nothing (clean output) |
| `genie` | Project Genie | Bottom-centre watermark |

## Requirements

```
torch
transformers
sam2  # https://github.com/facebookresearch/sam2
opencv-python-headless
Pillow
huggingface_hub
```

GPU is strongly recommended. SAM 2 video tracking on 1000+ frames is
memory-intensive; an H100 or L40S handles it comfortably.
