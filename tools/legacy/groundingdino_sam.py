#!/usr/bin/env python3
"""GroundingDINO + SAM 2: text-conditioned detection with pixel-level segmentation.

Pipeline: GroundingDINO detects objects from a text prompt (providing bounding boxes
and phrase labels) -> SAM 2 segments each detected region at pixel level.

Text conditioning / distinguishing different objects is achieved through
GroundingDINO's phrase labels: each segmented mask inherits the phrase from
its detection box, and phrases receive deterministic per-label colours.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict as gdino_predict

from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ensure_device(device: str | None = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_frame_bgr(frame_bgr: np.ndarray) -> torch.Tensor:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    image_tensor, _ = transform(image_pil, None)
    return image_tensor


# ---------------------------------------------------------------------------
# GroundingDINO detection (identical to groundingdino_bytetrack.py)
# ---------------------------------------------------------------------------

def grounding_dino_detect(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    h, w = frame_bgr.shape[:2]
    image = preprocess_frame_bgr(frame_bgr).to(device)

    with torch.no_grad():
        boxes, logits, phrases = gdino_predict(
            model=model,
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )

    if not isinstance(boxes, torch.Tensor) or boxes.numel() == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
        )

    scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
    boxes_abs = boxes * scale
    boxes_xyxy = box_convert(boxes=boxes_abs, in_fmt="cxcywh", out_fmt="xyxy")
    boxes_xyxy = boxes_xyxy.detach().cpu().numpy().astype(np.float32)

    if isinstance(logits, torch.Tensor):
        scores = logits.detach().cpu().numpy().astype(np.float32).reshape(-1)
    else:
        scores = np.asarray(logits, dtype=np.float32).reshape(-1)

    phrases = [str(p) for p in phrases]
    n = min(len(boxes_xyxy), len(scores), len(phrases))
    return boxes_xyxy[:n], scores[:n], phrases[:n]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def color_for_phrase(phrase: str) -> Tuple[int, int, int]:
    """Deterministic colour per phrase so the same label is consistent across frames."""
    seed = hash(phrase) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    c = rng.integers(80, 230, size=3).tolist()
    return int(c[0]), int(c[1]), int(c[2])


def draw_mask_overlay(
    frame: np.ndarray,
    mask_bool: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.4,
) -> None:
    if not np.any(mask_bool):
        return
    color_arr = np.array(color, dtype=np.float32)
    region = frame[mask_bool].astype(np.float32)
    blended = region * (1.0 - alpha) + color_arr * alpha
    frame[mask_bool] = np.clip(blended, 0, 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(frame, contours, -1, color, 2, cv2.LINE_AA)


def draw_box(
    frame: np.ndarray,
    box_xyxy: np.ndarray,
    label: str,
    color: Tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = np.round(box_xyxy).astype(int).tolist()
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1] - 1, x2)
    y2 = min(frame.shape[0] - 1, y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_text = max(th + 4, y1)
    cv2.rectangle(
        frame,
        (x1, y_text - th - baseline - 4),
        (x1 + tw + 6, y_text + 2),
        color,
        thickness=-1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 3, y_text - baseline - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def should_take_frame(frame_idx: int, src_fps: float, viz_fps: float) -> bool:
    if viz_fps <= 0:
        return False
    if src_fps <= 0:
        return True
    stride = max(1, int(round(src_fps / viz_fps)))
    return (frame_idx % stride) == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="GroundingDINO + SAM 2: text-conditioned detection with pixel-level segmentation."
    )
    ap.add_argument("--video", required=True, help="Input .mp4 path")
    ap.add_argument(
        "--text_prompt", required=True,
        help="GroundingDINO prompt (use ' . ' to separate multiple object categories)",
    )
    ap.add_argument("--grounding_cfg", required=True, help="GroundingDINO config path")
    ap.add_argument("--grounding_ckpt", required=True, help="GroundingDINO checkpoint path")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")

    ap.add_argument("--box_threshold", type=float, default=0.35)
    ap.add_argument("--text_threshold", type=float, default=0.25)

    ap.add_argument(
        "--sam_model", default="facebook/sam2.1-hiera-large",
        help="HuggingFace model ID for SAM 2 (checkpoint auto-downloads)",
    )
    ap.add_argument(
        "--mask_alpha", type=float, default=0.4,
        help="Mask overlay transparency (0 = transparent, 1 = opaque)",
    )

    ap.add_argument("--max_frames", type=int, default=None, help="Max source frames to read")
    ap.add_argument("--viz_fps", type=float, default=5.0, help="FPS at which to save annotated frames")
    ap.add_argument("--output_dir", default="segmentation_frames", help="Directory for annotated frames")
    ap.add_argument("--image_ext", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--output_json", default="grounding_sam_report.json", help="JSON report path")

    ap.add_argument(
        "--draw_boxes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw bounding boxes alongside masks (--no-draw_boxes to disable)",
    )

    args = ap.parse_args()
    device = ensure_device(args.device)

    # -- Load GroundingDINO ------------------------------------------------
    print("Loading GroundingDINO ...")
    gdino_model = load_model(
        model_config_path=args.grounding_cfg,
        model_checkpoint_path=args.grounding_ckpt,
        device=device,
    ).to(device)
    gdino_model.eval()

    # -- Load SAM 2 --------------------------------------------------------
    print(f"Loading SAM 2 ({args.sam_model}) ...")
    sam2_model = build_sam2_hf(args.sam_model, device=torch.device(device))
    sam_predictor = SAM2ImagePredictor(sam_model=sam2_model)
    print("SAM 2 ready.")

    # -- Open video --------------------------------------------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_pixels = height * width

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "video": args.video,
        "text_prompt": args.text_prompt,
        "device": device,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "sam_model": args.sam_model,
        "src_fps": src_fps,
        "viz_fps": args.viz_fps,
        "frame_size": [height, width],
        "saved_frames": [],
        "phrase_summary": {},
    }

    frame_idx = 0
    saved_count = 0
    use_autocast = device == "cuda"

    while True:
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if not should_take_frame(frame_idx, src_fps, args.viz_fps):
            frame_idx += 1
            continue

        # -- Detect --------------------------------------------------------
        boxes_xyxy, scores, phrases = grounding_dino_detect(
            model=gdino_model,
            frame_bgr=frame,
            caption=args.text_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=device,
        )

        det_records: List[Dict[str, Any]] = []

        if len(boxes_xyxy) > 0:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ac = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if use_autocast
                else nullcontext()
            )

            with torch.inference_mode(), ac:
                sam_predictor.set_image(image_rgb)

                for i, (box, score, phrase) in enumerate(
                    zip(boxes_xyxy, scores, phrases)
                ):
                    sam_masks, sam_scores, _ = sam_predictor.predict(
                        box=box[np.newaxis],
                        multimask_output=False,
                    )

                    if isinstance(sam_masks, torch.Tensor):
                        sam_masks = sam_masks.cpu().numpy()
                    if sam_masks.ndim == 4:
                        sam_masks = sam_masks[:, 0]
                    mask_bool = sam_masks[0].astype(bool)

                    color = color_for_phrase(phrase)
                    draw_mask_overlay(frame, mask_bool, color, args.mask_alpha)

                    if args.draw_boxes:
                        label = f"{phrase} {score:.2f}"
                        draw_box(frame, box, label, color)

                    mask_area = int(mask_bool.sum())
                    det_records.append(
                        {
                            "box_xyxy": [float(x) for x in box.tolist()],
                            "score": float(score),
                            "phrase": phrase,
                            "mask_area_px": mask_area,
                            "mask_fraction": (
                                mask_area / total_pixels if total_pixels > 0 else 0.0
                            ),
                        }
                    )

                    ps = report["phrase_summary"]
                    if phrase not in ps:
                        ps[phrase] = {
                            "total_detections": 0,
                            "first_frame": frame_idx,
                            "last_frame": frame_idx,
                        }
                    ps[phrase]["total_detections"] += 1
                    ps[phrase]["last_frame"] = frame_idx

        # -- Save annotated frame -----------------------------------------
        out_path = out_dir / f"frame_{frame_idx:06d}.{args.image_ext}"
        if args.image_ext == "jpg":
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            cv2.imwrite(str(out_path), frame)

        report["saved_frames"].append(
            {
                "frame_index": frame_idx,
                "time_sec": (frame_idx / src_fps) if src_fps > 0 else None,
                "image_path": str(out_path),
                "num_detections": len(det_records),
                "detections": det_records,
            }
        )

        saved_count += 1
        frame_idx += 1

    cap.release()

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved {saved_count} annotated frames to: {out_dir}")
    print(f"Wrote JSON report to: {out_json}")


if __name__ == "__main__":
    main()
