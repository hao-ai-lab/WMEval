# WMEval/evaluation/modules/_liquid_common.py
"""Shared helpers for the LiquidContainer eval modules.

This module provides:
  - Schema dataclasses (LiquidMaskResult, PIVField, FillCalibration)
  - Pure-numerical helpers (divergence_curl_2d, fit_fill_calibration,
    estimate_fill_curve_from_mask, detect_pour_onset_from_curve)
  - Cached I/O wrappers around SAM2 (get_liquid_mask_video) and
    OpenPIV (get_piv_field)

The cache key is (video_path, content_hash, config_hash) where config_hash
covers only the keys that affect each output. Changing PIV settings does
not invalidate SAM2 caches.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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


@dataclass
class FillCalibration:
    """Linear fit mask_area_px → fill_fraction in [0, 1]."""

    slope: float
    intercept: float
    r2: float
    method: str = "linear_robust"


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

    # Central differences on last two axes.
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

    # Mask: a pixel is valid if itself AND its left/right/up/down neighbours are valid.
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


def _fit_linear_robust(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Robust linear fit using repeated-median (Siegel) slopes.

    Returns (slope, intercept, r2). Siegel's estimator is breakdown-resistant
    to up to 50% contaminated data, which is what the plan's "linear_robust"
    contract requires; the discretization bias from rasterized fill rectangles
    breaks an OLS+trim implementation against the test's <1e-3 intercept band.
    """
    from scipy.stats import siegelslopes

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"_fit_linear_robust: shape mismatch x={x.shape} y={y.shape}")
    if x.size < 2:
        raise ValueError("_fit_linear_robust: need at least 2 points")

    result = siegelslopes(y, x)
    slope = float(result.slope)
    intercept = float(result.intercept)

    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return slope, intercept, r2


def _mask_areas(masks: np.ndarray) -> np.ndarray:
    """Per-frame mask area in pixels."""
    if masks.ndim != 3:
        raise ValueError(f"_mask_areas: expected (T, H, W) got {masks.shape}")
    return masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.float64)


def fit_fill_calibration(
    *,
    unity_masks: np.ndarray,
    unity_log_fill: np.ndarray,
) -> FillCalibration:
    """Fit mask_area_px → fill_fraction on the Unity reference.

    The Unity log gives ground-truth fill[t]. The Unity video gives mask[t].
    We fit a robust linear model so the same calibration can be applied
    to the WM-side mask sequence (assuming same camera framing).
    """
    if unity_masks.shape[0] != unity_log_fill.shape[0]:
        raise ValueError(
            f"fit_fill_calibration: T mismatch masks={unity_masks.shape[0]} "
            f"fill={unity_log_fill.shape[0]}"
        )
    areas = _mask_areas(unity_masks)
    slope, intercept, r2 = _fit_linear_robust(areas, unity_log_fill)
    return FillCalibration(slope=slope, intercept=intercept, r2=r2, method="linear_robust")


def estimate_fill_curve_from_mask(
    masks: np.ndarray,
    *,
    calibration: FillCalibration,
) -> np.ndarray:
    """Apply a fitted calibration to a (possibly WM-side) mask sequence."""
    areas = _mask_areas(masks)
    fill = calibration.slope * areas + calibration.intercept
    return np.clip(fill.astype(np.float32), 0.0, 1.0)


def _moving_average(curve: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return curve.astype(np.float32, copy=False)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad = window // 2
    padded = np.pad(curve, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(curve)].astype(np.float32)


def detect_pour_onset_from_curve(
    curve: np.ndarray,
    fps: float,
    *,
    threshold_drop: float = 0.005,
    k_consecutive: int = 5,
    smoothing_window: int = 1,
) -> Optional[float]:
    """Return the time (sec) of the first sustained downward drop in a fill curve.

    A frame t is "dropping" if smoothed_curve[t] - smoothed_curve[t-1] < -threshold_drop.
    The onset is the first index of a run of >= k_consecutive dropping frames.
    Returns None if no such run exists.
    """
    if fps <= 0:
        raise ValueError("detect_pour_onset_from_curve: fps must be positive")
    curve = np.asarray(curve, dtype=np.float32).reshape(-1)
    if curve.size < k_consecutive + 1:
        return None
    smoothed = _moving_average(curve, smoothing_window)
    deltas = np.diff(smoothed)
    dropping = deltas < -float(threshold_drop)

    # Find the first run of length >= k_consecutive of `dropping == True`.
    run_len = 0
    for i, is_drop in enumerate(dropping):
        if is_drop:
            run_len += 1
            if run_len >= k_consecutive:
                # Onset corresponds to the start of the run, which is at index
                # (i - k_consecutive + 1) in `dropping`, i.e. between curve[i-k+1] and curve[i-k+2].
                onset_frame = i - k_consecutive + 2
                return float(onset_frame) / float(fps)
        else:
            run_len = 0
    return None


@dataclass
class UnityLog:
    """Parsed `unity_gt_log.json` payload, exposing per-frame arrays."""

    scene: str
    fps: float
    bottle_local_volume_ml: float
    pour_threshold_deg: float
    k_pour: float
    initial_fill_fraction: float

    # Per-frame arrays, all shape (T,) unless noted.
    t_sim: np.ndarray
    fill_fraction: np.ndarray
    surface_height: np.ndarray
    pour_rate: np.ndarray
    spilled_volume: np.ndarray
    is_pouring: np.ndarray
    tilt_deg: np.ndarray
    bottle_position: np.ndarray  # (T, 3)
    bottle_rotation_quat: np.ndarray  # (T, 4)
    actions: np.ndarray  # (T, 6) — order: fwd, right, up, pitch, yaw, roll
    camera_position: np.ndarray  # (T, 3)
    camera_rotation_quat: np.ndarray  # (T, 4)
    camera_intrinsics: list  # length T, each dict {fx, fy, cx, cy, near, far}

    @property
    def num_frames(self) -> int:
        return int(self.t_sim.shape[0])


_REQUIRED_HEADER_KEYS = (
    "scene",
    "fps",
    "bottle_local_volume_ml",
    "pour_threshold_deg",
    "k_pour",
    "initial_fill_fraction",
    "frames",
)
_REQUIRED_FRAME_KEYS = ("t_sim", "frame_index", "bottle", "actions", "liquid", "camera")
_ACTION_ORDER = ("fwd", "right", "up", "pitch", "yaw", "roll")


def load_unity_log(path: str | Path) -> UnityLog:
    """Parse the per-frame Unity ground-truth JSON written by LiquidGroundTruthRecorder.cs."""
    raw = json.loads(Path(path).read_text())
    for key in _REQUIRED_HEADER_KEYS:
        if key not in raw:
            raise ValueError(f"unity log {path}: missing required key: {key}")
    frames = raw["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"unity log {path}: 'frames' must be a non-empty list")
    for i, fr in enumerate(frames):
        for key in _REQUIRED_FRAME_KEYS:
            if key not in fr:
                raise ValueError(f"unity log {path}: frame {i} missing {key!r}")

    t_sim = np.array([f["t_sim"] for f in frames], dtype=np.float64)
    fill_fraction = np.array([f["liquid"]["fill_fraction"] for f in frames], dtype=np.float32)
    surface_height = np.array([f["liquid"]["surface_height"] for f in frames], dtype=np.float32)
    pour_rate = np.array([f["liquid"]["pour_rate"] for f in frames], dtype=np.float32)
    spilled_volume = np.array([f["liquid"]["spilled_volume"] for f in frames], dtype=np.float32)
    is_pouring = np.array([bool(f["liquid"]["is_pouring"]) for f in frames], dtype=bool)
    tilt_deg = np.array([f["bottle"]["tilt_deg_from_up"] for f in frames], dtype=np.float32)
    bottle_position = np.array([f["bottle"]["position"] for f in frames], dtype=np.float32)
    bottle_rotation_quat = np.array([f["bottle"]["rotation_quat"] for f in frames], dtype=np.float32)
    actions = np.array(
        [[f["actions"][k] for k in _ACTION_ORDER] for f in frames], dtype=np.float32
    )
    camera_position = np.array([f["camera"]["position"] for f in frames], dtype=np.float32)
    camera_rotation_quat = np.array([f["camera"]["rotation_quat"] for f in frames], dtype=np.float32)
    camera_intrinsics = [f["camera"].get("intrinsics", {}) for f in frames]

    return UnityLog(
        scene=str(raw["scene"]),
        fps=float(raw["fps"]),
        bottle_local_volume_ml=float(raw["bottle_local_volume_ml"]),
        pour_threshold_deg=float(raw["pour_threshold_deg"]),
        k_pour=float(raw["k_pour"]),
        initial_fill_fraction=float(raw["initial_fill_fraction"]),
        t_sim=t_sim,
        fill_fraction=fill_fraction,
        surface_height=surface_height,
        pour_rate=pour_rate,
        spilled_volume=spilled_volume,
        is_pouring=is_pouring,
        tilt_deg=tilt_deg,
        bottle_position=bottle_position,
        bottle_rotation_quat=bottle_rotation_quat,
        actions=actions,
        camera_position=camera_position,
        camera_rotation_quat=camera_rotation_quat,
        camera_intrinsics=camera_intrinsics,
    )


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

    # Lazy import — keeps unit-test path SAM2-free.
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
    # Standard luma weights. Avoids importing cv2 again.
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

    # Lazy import — keeps unit tests OpenPIV-free.
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
        # Window center grid for masking.
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
