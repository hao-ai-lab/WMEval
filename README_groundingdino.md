# GroundingDINO Tracking/Segmentation (Mini README)

This folder contains copied GroundingDINO-related scripts from `VideoScience/judge/cv_tools`.
They are organized as:

- `tools/`: Python pipelines (`groundingdino_bytetrack.py`, `groundingdino_sam.py`)
- `scripts/`: runnable shell examples (`groundingdino_bytetrack*.sh`, `groundingdino_sam*.sh`)

## Requirements

- Python 3.10+
- PyTorch + CUDA (optional but recommended)
- OpenCV (`cv2`), NumPy, Pillow, torchvision
- GroundingDINO package/codebase available in Python path
- `cjm_byte_track` for ByteTrack script
- SAM 2 package (`sam2`) for segmentation script

## Quick usage

ByteTrack pipeline:

```bash
python tools/groundingdino_bytetrack.py \
  --video /path/to/video.mp4 \
  --text_prompt "ball . pendulum" \
  --grounding_cfg /path/to/GroundingDINO_SwinT_OGC.py \
  --grounding_ckpt /path/to/groundingdino_swint_ogc.pth \
  --output_dir mechanics_tracking_frames/run1 \
  --output_json mechanics_tracking_frames/run1/reports/tracking_report.json
```

SAM 2 pipeline:

```bash
python tools/groundingdino_sam.py \
  --video /path/to/video.mp4 \
  --text_prompt "ball . car" \
  --grounding_cfg /path/to/GroundingDINO_SwinT_OGC.py \
  --grounding_ckpt /path/to/groundingdino_swint_ogc.pth \
  --output_dir mechanics_tracking_frames/run1/sam \
  --output_json mechanics_tracking_frames/run1/reports/sam_report.json
```

Use the scripts in `scripts/` as ready-to-edit templates for concrete runs, for example:

```bash
bash scripts/groundingdino_bytetrack.sh
bash scripts/groundingdino_sam.sh
```
