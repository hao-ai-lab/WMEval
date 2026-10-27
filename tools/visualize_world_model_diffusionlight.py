#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

try:
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception:
    build_sam2_hf = None
    SAM2ImagePredictor = None


def ensure_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def sample_frames(video_path: str, eval_fps: float, max_frames: Optional[int]) -> Tuple[List[np.ndarray], List[int], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    stride = 1 if src_fps <= 0 or eval_fps <= 0 else max(1, int(round(src_fps / eval_fps)))

    frames: List[np.ndarray] = []
    indices: List[int] = []
    idx = 0
    while True:
        if max_frames is not None and len(frames) >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(frame)
            indices.append(idx)
        idx += 1

    cap.release()
    return frames, indices, src_fps


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 1] = 255
    norm = np.clip((mag / (np.percentile(mag, 99) + 1e-6)) * 255.0, 0, 255)
    hsv[..., 2] = norm.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _to_rgb_tchw(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    frame_rgb = frame_bgr[..., ::-1].copy()
    return (torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0).to(device)


def raft_flow(model: torch.nn.Module, transform, prev_bgr: np.ndarray, next_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    t0 = _to_rgb_tchw(prev_bgr, device)
    t1 = _to_rgb_tchw(next_bgr, device)
    t0, t1 = transform(t0, t1)
    h, w = t0.shape[-2:]
    target_h = max(128, int(np.ceil(h / 8.0) * 8))
    target_w = max(128, int(np.ceil(w / 8.0) * 8))
    if target_h != h or target_w != w:
        t0 = F.interpolate(t0, size=(target_h, target_w), mode="bilinear", align_corners=False)
        t1 = F.interpolate(t1, size=(target_h, target_w), mode="bilinear", align_corners=False)
    with torch.no_grad():
        flow = model(t0, t1)[-1][0:1]
    if flow.shape[-2:] != (h, w):
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
        flow[:, 0] *= float(w) / float(target_w)
        flow[:, 1] *= float(h) / float(target_h)
    return flow[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)


def light_from_hdr(hdr_path: Path) -> Tuple[float, float]:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    hdr = iio.imread(str(hdr_path), plugin="opencv")
    lum = 0.2126 * hdr[..., 2] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 0]
    h, w = lum.shape
    y, x = np.unravel_index(np.argmax(lum), lum.shape)
    az = (x / max(w - 1, 1)) * 360.0 - 180.0
    el = (0.5 - y / max(h - 1, 1)) * 180.0
    return float(az), float(el)


def light_proxy_from_frame(frame_bgr: np.ndarray) -> Tuple[float, float, Tuple[int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    th = np.percentile(gray, 99.3)
    mask = gray >= th
    if not np.any(mask):
        y, x = np.unravel_index(np.argmax(gray), gray.shape)
    else:
        ys, xs = np.where(mask)
        weights = gray[mask].astype(np.float64) + 1e-6
        x = int(np.sum(xs * weights) / np.sum(weights))
        y = int(np.sum(ys * weights) / np.sum(weights))
    h, w = gray.shape
    az = (x / max(w - 1, 1)) * 360.0 - 180.0
    el = (1.0 - y / max(h - 1, 1)) * 180.0 - 90.0
    return float(az), float(el), (int(x), int(y))


def shadow_proxy_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)
    h, w = v.shape
    m = (v <= np.percentile(v, 35.0)) & (s <= np.percentile(s, 75.0))
    bottom = np.zeros((h, w), dtype=bool)
    bottom[int(h * 0.35) :, :] = True
    m = m & bottom
    u8 = (m.astype(np.uint8) * 255)
    k = np.ones((5, 5), np.uint8)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, k)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k)
    return u8 > 0


def bbox_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize lighting, RAFT flow, and shadow extraction on one video")
    ap.add_argument("--video", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--eval_fps", type=float, default=2.0)
    ap.add_argument("--max_frames", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--diffusionlight_hdr_dir", default=None, help="Optional precomputed DiffusionLight HDR dir")
    ap.add_argument("--sam2_model", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--disable_sam2", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    frames, source_indices, src_fps = sample_frames(args.video, args.eval_fps, args.max_frames)
    if not frames:
        raise RuntimeError("No frames sampled")

    device = torch.device(ensure_device(args.device))
    raft_weights = Raft_Large_Weights.DEFAULT
    raft_transform = raft_weights.transforms()
    raft_model = raft_large(weights=raft_weights, progress=False).to(device).eval()

    sam2_predictor = None
    if (not args.disable_sam2) and build_sam2_hf is not None and SAM2ImagePredictor is not None:
        sam2_model = build_sam2_hf(args.sam2_model, device=device)
        sam2_predictor = SAM2ImagePredictor(sam_model=sam2_model)

    hdr_files: List[Path] = []
    if args.diffusionlight_hdr_dir:
        p = Path(args.diffusionlight_hdr_dir)
        hdr_files = sorted(p.glob("*.exr")) or sorted(p.glob("*.hdr"))
        if len(hdr_files) > len(frames):
            idx = np.linspace(0, len(hdr_files) - 1, num=len(frames))
            hdr_files = [hdr_files[int(round(i))] for i in idx]

    shadow_bbox = None
    if sam2_predictor is not None:
        init_mask = shadow_proxy_mask(frames[0])
        shadow_bbox = bbox_from_mask(init_mask)
        if shadow_bbox is None:
            sam2_predictor = None

    flow_mags: List[float] = []
    light_track: List[Dict[str, Any]] = []
    shadow_area: List[int] = []
    rendered: List[np.ndarray] = []

    for i, frame in enumerate(frames):
        vis = frame.copy()
        h, w = vis.shape[:2]

        if i == 0:
            flow = np.zeros((h, w, 2), dtype=np.float32)
        else:
            flow = raft_flow(raft_model, raft_transform, frames[i - 1], frame, device)
        flow_mags.append(float(np.linalg.norm(flow, axis=2).mean()))
        flow_color = flow_to_color(flow)
        vis = cv2.addWeighted(vis, 0.65, flow_color, 0.35, 0.0)

        if hdr_files and i < len(hdr_files):
            try:
                az, el = light_from_hdr(hdr_files[i])
                lx = int((az + 180.0) / 360.0 * (w - 1))
                ly = int((1.0 - (el + 90.0) / 180.0) * (h - 1))
                light_mode = "diffusionlight"
            except Exception:
                az, el, (lx, ly) = light_proxy_from_frame(frame)
                light_mode = "brightest_proxy_fallback"
        else:
            az, el, (lx, ly) = light_proxy_from_frame(frame)
            light_mode = "brightest_proxy"
        light_track.append({"frame": i, "azimuth_deg": az, "elevation_deg": el})

        cv2.circle(vis, (lx, ly), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.line(vis, (w // 2, h // 2), (lx, ly), (0, 255, 255), 2, cv2.LINE_AA)

        if sam2_predictor is not None and shadow_bbox is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sam2_predictor.set_image(rgb)
            with torch.inference_mode():
                pred, _, _ = sam2_predictor.predict(box=shadow_bbox[None, :], multimask_output=False)
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            if pred.ndim == 4:
                pred = pred[:, 0]
            mask = pred[0].astype(bool)
            next_bbox = bbox_from_mask(mask)
            if next_bbox is not None:
                shadow_bbox = next_bbox
        else:
            mask = shadow_proxy_mask(frame)

        shadow_area.append(int(mask.sum()))
        overlay = np.zeros_like(vis)
        overlay[mask] = (60, 220, 60)
        vis = cv2.addWeighted(vis, 1.0, overlay, 0.35, 0.0)

        cv2.putText(vis, f"frame={i} src_idx={source_indices[i]}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(vis, f"light={light_mode} az={az:.1f} el={el:.1f}", (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(vis, f"flow_mean={flow_mags[-1]:.3f} shadow_area={shadow_area[-1]}", (6, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imwrite(str(frame_dir / f"frame_{i:04d}.jpg"), vis)
        rendered.append(vis)

    out_video = out_dir / "visualization.mp4"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), max(args.eval_fps, 1.0), (rendered[0].shape[1], rendered[0].shape[0]))
    for f in rendered:
        writer.write(f)
    writer.release()

    summary = {
        "video": args.video,
        "num_frames": len(frames),
        "source_fps": src_fps,
        "eval_fps": args.eval_fps,
        "device": str(device),
        "flow_mean_mag": float(np.mean(np.asarray(flow_mags))),
        "shadow_area_mean": float(np.mean(np.asarray(shadow_area))),
        "light_track": light_track,
        "visualization_video": str(out_video),
        "frames_dir": str(frame_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
