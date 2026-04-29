# WMEval/evaluation/modules/_liquid_common.py
"""Shared helpers for the LiquidContainer eval modules.

This module provides:
  - Schema dataclasses (LiquidMaskResult, PIVField)
  - Pure-numerical helpers (divergence_curl_2d, _mask_areas)
  - Cached I/O wrappers around SAM2 (get_liquid_mask_video) and
    OpenPIV (get_piv_field)

The cache key is (video_path, content_hash, config_hash) where config_hash
covers only the keys that affect each output. Changing PIV settings does
not invalidate SAM2 caches.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LiquidMaskResult:
    """Per-frame binary masks for the liquid region inside the bottle.

    masks[t, h, w] is True where the liquid (NOT including the glass / bottle)
    occupies the pixel.
    """

    masks: np.ndarray  # (T, H, W) bool
    fps: float
    sam2_model: str
    prompt: str

    @property
    def num_frames(self) -> int:
        return int(self.masks.shape[0])


@dataclass
class PIVField:
    """2D velocity field (u right-positive, v down-positive in image coords).

    For an OpenPIV computation with window size W and overlap O, the spatial
    grid is downsampled to (T, H//(W-O), W//(W-O)). For convenience we keep
    these as the field's H, W (they will be smaller than image H, W).
    """

    u: np.ndarray  # (..., H, W) float32, px/frame
    v: np.ndarray  # (..., H, W) float32, px/frame
    valid: np.ndarray  # (..., H, W) bool — False where PIV failed


def divergence_curl_2d(field: PIVField) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference 2D divergence and curl on the (H, W) plane.

    Operates on the last two axes if u, v have a leading time axis. Invalid
    pixels (and any neighbour participating in the central-difference stencil
    that is invalid) propagate as NaN. The borders are also NaN.
    """
    u = field.u.astype(np.float32, copy=False)
    v = field.v.astype(np.float32, copy=False)
    valid = field.valid

    if u.shape != v.shape or u.shape != valid.shape:
        raise ValueError(
            f"divergence_curl_2d: shape mismatch u={u.shape} v={v.shape} valid={valid.shape}"
        )

    du_dx = np.full_like(u, np.nan)
    du_dy = np.full_like(u, np.nan)
    dv_dx = np.full_like(v, np.nan)
    dv_dy = np.full_like(v, np.nan)

    du_dx[..., :, 1:-1] = 0.5 * (u[..., :, 2:] - u[..., :, :-2])
    du_dy[..., 1:-1, :] = 0.5 * (u[..., 2:, :] - u[..., :-2, :])
    dv_dx[..., :, 1:-1] = 0.5 * (v[..., :, 2:] - v[..., :, :-2])
    dv_dy[..., 1:-1, :] = 0.5 * (v[..., 2:, :] - v[..., :-2, :])

    div = du_dx + dv_dy
    curl = dv_dx - du_dy

    stencil_ok = np.zeros_like(valid)
    stencil_ok[..., 1:-1, 1:-1] = (
        valid[..., 1:-1, 1:-1]
        & valid[..., 1:-1, :-2]
        & valid[..., 1:-1, 2:]
        & valid[..., :-2, 1:-1]
        & valid[..., 2:, 1:-1]
    )
    div = np.where(stencil_ok, div, np.nan)
    curl = np.where(stencil_ok, curl, np.nan)
    return div, curl


def _mask_areas(masks: np.ndarray) -> np.ndarray:
    """Per-frame mask area in pixels."""
    if masks.ndim != 3:
        raise ValueError(f"_mask_areas: expected (T, H, W) got {masks.shape}")
    return masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.float64)


def _content_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()[:16]


def _cache_path(cache_dir: Path, video_path: str | Path, kind: str, config_hash: str) -> Path:
    ch = _content_hash(video_path)
    return cache_dir / f"{Path(video_path).stem}.{kind}.{ch}.{config_hash}.pkl"


def _config_hash(config: dict) -> str:
    items = sorted((str(k), str(v)) for k, v in config.items())
    return hashlib.sha256(repr(items).encode("utf-8")).hexdigest()[:16]


def _read_video_frames_uint8(path: str, max_frames: int | None = None) -> np.ndarray:
    """Decode the entire video into a (T, H, W, 3) uint8 RGB array."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"_read_video_frames_uint8: no frames read from {path}")
    return np.stack(frames).astype(np.uint8)


def get_liquid_mask_video(
    *,
    video_path: str,
    sam2_model: str,
    prompt: str,
    fps: float,
    max_pairs: int | None,
    cache_dir: str | Path,
) -> LiquidMaskResult:
    """Run SAM2 video predictor on `video_path`, prompted by `prompt`, returning per-frame masks.

    Caches the binary mask volume under cache_dir keyed on (video content hash, sam2 settings).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = _config_hash({"sam2_model": sam2_model, "prompt": prompt, "fps": fps, "max_pairs": max_pairs})
    cache_file = _cache_path(cache_dir, video_path, "sam2_mask", cfg)
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    from sam2.build_sam import build_sam2_video_predictor
    import torch

    frames = _read_video_frames_uint8(video_path, max_frames=max_pairs)
    predictor = build_sam2_video_predictor(sam2_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor.to(device)

    state = predictor.init_state(video_path=video_path)
    text_input = prompt.strip() or "the liquid inside the bottle"
    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=0,
        labels=None,
        points=None,
        text=text_input,
    )
    masks: list[np.ndarray] = [np.zeros(frames.shape[1:3], dtype=bool)] * frames.shape[0]
    for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
        m = (mask_logits[0] > 0).cpu().numpy()
        if m.ndim == 3:
            m = m[0]
        masks[frame_idx] = m.astype(bool)

    result = LiquidMaskResult(
        masks=np.stack(masks).astype(bool), fps=float(fps), sam2_model=sam2_model, prompt=text_input
    )
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    return result


def _gray_uint8_video(video_path: str, max_frames: int | None) -> np.ndarray:
    rgb = _read_video_frames_uint8(video_path, max_frames=max_frames)
    g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
    return g  # (T, H, W)


def get_piv_field(
    *,
    video_path: str,
    mask: LiquidMaskResult,
    window_size: int,
    overlap: int,
    fps: float,
    cache_dir: str | Path,
    max_frames: int | None = None,
) -> PIVField:
    """OpenPIV cross-correlation per consecutive-frame pair, restricted to the
    intersection of consecutive masks. Returns u, v, valid arrays of shape
    (T-1, H_p, W_p) where (H_p, W_p) is the PIV grid.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = _config_hash(
        {
            "window_size": window_size,
            "overlap": overlap,
            "fps": fps,
            "max_frames": max_frames,
            "sam2_model": mask.sam2_model,
            "prompt": mask.prompt,
        }
    )
    cache_file = _cache_path(cache_dir, video_path, "piv", cfg)
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    from openpiv import pyprocess

    gray = _gray_uint8_video(video_path, max_frames=max_frames)
    if gray.shape[0] < 2:
        raise RuntimeError(f"get_piv_field: need ≥2 frames, got {gray.shape[0]}")

    masks = mask.masks
    if masks.shape[0] < gray.shape[0]:
        raise ValueError(
            f"get_piv_field: mask has {masks.shape[0]} frames, video has {gray.shape[0]}"
        )

    us: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    for t in range(gray.shape[0] - 1):
        a = gray[t].astype(np.int32)
        b = gray[t + 1].astype(np.int32)
        u, v, sig2noise = pyprocess.extended_search_area_piv(
            a,
            b,
            window_size=int(window_size),
            overlap=int(overlap),
            dt=1.0,
            search_area_size=int(window_size),
            sig2noise_method="peak2peak",
        )
        x_centers, y_centers = pyprocess.get_coordinates(
            image_size=a.shape, search_area_size=int(window_size), overlap=int(overlap)
        )
        joint_mask = masks[t] & masks[t + 1]
        sampled = joint_mask[
            np.clip(y_centers.astype(int), 0, joint_mask.shape[0] - 1),
            np.clip(x_centers.astype(int), 0, joint_mask.shape[1] - 1),
        ]
        valid = sampled & (sig2noise > 1.05) & np.isfinite(u) & np.isfinite(v)
        us.append(np.where(valid, u, np.nan).astype(np.float32))
        vs.append(np.where(valid, v, np.nan).astype(np.float32))
        valids.append(valid)

    field = PIVField(u=np.stack(us), v=np.stack(vs), valid=np.stack(valids))
    with open(cache_file, "wb") as f:
        pickle.dump(field, f)
    return field
