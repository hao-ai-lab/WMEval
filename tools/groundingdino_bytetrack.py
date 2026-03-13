#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict as gdino_predict

from cjm_byte_track.core import BYTETracker
from cjm_byte_track.matching import match_detections_with_tracks


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

    # normalized cxcywh -> absolute xyxy
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


def color_for_track(track_id: int) -> Tuple[int, int, int]:
    if track_id < 0:
        return (180, 180, 180)
    rng = np.random.default_rng(track_id)
    c = rng.integers(64, 255, size=3).tolist()
    return int(c[0]), int(c[1]), int(c[2])


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


def draw_trajectory(
    frame: np.ndarray,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    max_points: Optional[int] = None,
) -> None:
    pts = points if max_points is None else points[-max_points:]
    if len(pts) >= 2:
        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [arr], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
    for i, pt in enumerate(pts):
        radius = 3 if i < len(pts) - 1 else 5
        cv2.circle(frame, pt, radius, color, -1, lineType=cv2.LINE_AA)


def should_take_frame(frame_idx: int, src_fps: float, viz_fps: float) -> bool:
    """
    Keep frames so that we process approx viz_fps frames per second.
    Uses a simple stride. Example: src=30, viz=5 -> stride=6 -> take every 6 frames.
    """
    if viz_fps <= 0:
        return False
    if src_fps <= 0:
        # if fps unknown, fall back to taking every frame
        return True
    stride = max(1, int(round(src_fps / viz_fps)))
    return (frame_idx % stride) == 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="GroundingDINO + ByteTrack; save annotated tracking frames at a target visualization FPS."
    )
    ap.add_argument("--video", required=True, help="Input .mp4 path")
    ap.add_argument("--text_prompt", required=True, help="GroundingDINO prompt")
    ap.add_argument("--grounding_cfg", required=True, help="GroundingDINO config path")
    ap.add_argument("--grounding_ckpt", required=True, help="GroundingDINO checkpoint path")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")

    ap.add_argument("--box_threshold", type=float, default=0.35)
    ap.add_argument("--text_threshold", type=float, default=0.25)

    ap.add_argument("--track_thresh", type=float, default=0.5)
    ap.add_argument("--track_buffer", type=int, default=30)
    ap.add_argument("--match_thresh", type=float, default=0.8)

    ap.add_argument("--max_frames", type=int, default=None, help="Max frames read from source video (by index)")
    ap.add_argument("--viz_fps", type=float, default=5.0, help="FPS at which to SAVE annotated frames")
    ap.add_argument("--output_dir", default="tracking_frames", help="Directory to write annotated frames")
    ap.add_argument("--image_ext", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--output_json", default="grounding_bytetrack_report.json", help="JSON report path")

    ap.add_argument(
        "--draw_untracked",
        action="store_true",
        help="Also draw detections that were not matched to a track",
    )
    ap.add_argument(
        "--trail_length",
        type=int,
        default=None,
        help="Max trajectory points to draw per track (None = full history)",
    )

    args = ap.parse_args()
    device = ensure_device(args.device)

    # Load model
    model = load_model(
        model_config_path=args.grounding_cfg,
        model_checkpoint_path=args.grounding_ckpt,
        device=device,
    ).to(device)
    model.eval()

    # Video reader
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Tracker (keep frame_rate as src fps if known)
    tracker = BYTETracker(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        frame_rate=(src_fps if src_fps > 0 else 30.0),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "video": args.video,
        "text_prompt": args.text_prompt,
        "device": device,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "track_thresh": args.track_thresh,
        "track_buffer": args.track_buffer,
        "match_thresh": args.match_thresh,
        "src_fps": src_fps,
        "viz_fps": args.viz_fps,
        "frame_size": [height, width],
        "saved_frames": [],          # only frames we saved
        "track_summary": {},
    }

    track_history: Dict[int, List[Tuple[int, int]]] = {}

    frame_idx = 0
    saved_count = 0

    while True:
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        take = should_take_frame(frame_idx, src_fps, args.viz_fps)
        if not take:
            frame_idx += 1
            continue

        # Run detection
        boxes_xyxy, scores, phrases = grounding_dino_detect(
            model=model,
            frame_bgr=frame,
            caption=args.text_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=device,
        )

        track_ids: List[int] = [-1] * len(boxes_xyxy)

        if len(boxes_xyxy) > 0:
            dets_for_tracker = np.concatenate([boxes_xyxy, scores[:, None]], axis=1).astype(np.float32)

            img_wh = (frame.shape[1], frame.shape[0])
            tracks = tracker.update(
                output_results=dets_for_tracker,
                img_info=img_wh,
                img_size=img_wh,
            )

            if len(tracks) > 0:
                track_ids = match_detections_with_tracks(
                    tlbr_boxes=boxes_xyxy.copy(),
                    track_ids=track_ids,
                    tracks=tracks,
                )

        # Draw and record
        det_records: List[Dict[str, Any]] = []
        for box, score, phrase, track_id in zip(boxes_xyxy, scores, phrases, track_ids):
            det_records.append(
                {
                    "box_xyxy": [float(x) for x in box.tolist()],
                    "score": float(score),
                    "phrase": phrase,
                    "track_id": int(track_id),
                }
            )

            if track_id != -1 or args.draw_untracked:
                label = f"id={track_id} {phrase} {score:.2f}" if track_id != -1 else f"{phrase} {score:.2f}"
                draw_box(frame, box, label, color_for_track(int(track_id)))

            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            tid_key = int(track_id)
            track_history.setdefault(tid_key, []).append((cx, cy))

            if track_id != -1:
                key = str(int(track_id))
                if key not in report["track_summary"]:
                    report["track_summary"][key] = {
                        "frames_seen": 0,
                        "first_frame": frame_idx,
                        "last_frame": frame_idx,
                        "phrase_counts": {},
                    }
                report["track_summary"][key]["frames_seen"] += 1
                report["track_summary"][key]["last_frame"] = frame_idx
                pcs = report["track_summary"][key]["phrase_counts"]
                pcs[phrase] = pcs.get(phrase, 0) + 1

        for tid, pts in track_history.items():
            if tid == -1 and not args.draw_untracked:
                continue
            if len(pts) >= 2:
                draw_trajectory(frame, pts, color_for_track(tid), max_points=args.trail_length)

        # Save annotated frame image
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
