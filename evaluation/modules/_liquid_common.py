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


def require_matched_fps(context, *, module_name: str, tol: float = 0.1) -> float:
    """Return shared fps for a (reference, candidate) pair, or raise.

    Every liquid module multiplies pixel/frame quantities by fps to convert
    to physical units (pixels/sec, Hz). If the two videos disagree on fps
    the metric silently scales by ref_fps/wm_fps. The shell pipeline runs
    normalize_fps on both videos before invoking the eval; this guard fires
    only when callers bypass that wrapper.
    """
    ref_fps = float(context.reference.metadata.fps)
    wm_fps = float(context.candidate.metadata.fps)
    if abs(ref_fps - wm_fps) > tol:
        raise ValueError(
            f"{module_name}: reference fps ({ref_fps}) != candidate fps ({wm_fps}); "
            f"run WMEval/scripts/normalize_fps.sh on both videos before evaluation"
        )
    return ref_fps


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
class BottleTrace:
    """Per-frame bottle centroid + bbox, used for bulk-motion compensation.

    Lingbot renders camera-moves+bottle-centered while Unity renders
    static-camera+bottle-moves; without compensation the PIV signal inside
    the liquid mask is dominated by bottle bulk translation in Unity and
    not in lingbot, so the two are not directly comparable. We track the
    BOTTLE (not the liquid) and subtract its centroid velocity to bring
    both into a bottle-relative frame.
    """

    centroid_xy: np.ndarray  # (T, 2) float64, image px (x right, y down)
    bbox_xyxy: np.ndarray    # (T, 4) float64, image px (x0, y0, x1, y1)
    valid: np.ndarray        # (T,)   bool — False where SAM2 lost the bottle


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


_GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
_GDINO_BOX_THRESHOLD = 0.35
_GDINO_TEXT_THRESHOLD = 0.25
_GDINO_CACHE: dict[tuple[str, str], tuple] = {}


def _get_grounding_dino(model_id: str, device: str):
    key = (model_id, device)
    if key in _GDINO_CACHE:
        return _GDINO_CACHE[key]
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    _GDINO_CACHE[key] = (processor, model)
    return processor, model


def _grounding_dino_box_xyxy(
    frame_rgb: np.ndarray,
    text: str,
    *,
    device: str,
    model_id: str = _GDINO_MODEL_ID,
    box_threshold: float = _GDINO_BOX_THRESHOLD,
    text_threshold: float = _GDINO_TEXT_THRESHOLD,
) -> np.ndarray:
    """Highest-scoring (x1,y1,x2,y2) box for `text` in `frame_rgb` (RGB uint8)."""
    import torch
    from PIL import Image

    processor, model = _get_grounding_dino(model_id, device)
    pil = Image.fromarray(frame_rgb)
    # GroundingDINO expects lowercase, period-terminated category strings.
    caption = text.strip().lower()
    if not caption.endswith("."):
        caption = caption + "."
    inputs = processor(images=pil, text=caption, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    h, w = frame_rgb.shape[:2]
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(h, w)],
    )[0]
    boxes = results["boxes"]
    scores = results["scores"]
    if boxes.numel() == 0:
        raise RuntimeError(
            f"GroundingDINO found no detections for prompt {text!r} on frame 0 "
            f"(thresholds box={box_threshold}, text={text_threshold}). "
            f"Try rephrasing the prompt or lowering thresholds."
        )
    best = int(torch.argmax(scores).item())
    return boxes[best].detach().cpu().numpy().astype(np.float32)


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

    The text `prompt` is resolved to a frame-0 bounding box via GroundingDINO
    (HuggingFace `IDEA-Research/grounding-dino-tiny`); SAM2 propagates from that
    box. Caches the binary mask volume under cache_dir keyed on (video content
    hash, sam2 settings, grounding-dino settings).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = _config_hash({
        "sam2_model": sam2_model,
        "prompt": prompt,
        "fps": fps,
        "max_pairs": max_pairs,
        "gdino_model": _GDINO_MODEL_ID,
        "gdino_box_threshold": _GDINO_BOX_THRESHOLD,
        "gdino_text_threshold": _GDINO_TEXT_THRESHOLD,
    })
    cache_file = _cache_path(cache_dir, video_path, "sam2_mask", cfg)
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    from sam2.build_sam import build_sam2_video_predictor, build_sam2_video_predictor_hf
    import torch

    frames = _read_video_frames_uint8(video_path, max_frames=max_pairs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # `sam2_model` may be either a HuggingFace id (e.g. "facebook/sam2.1-hiera-small")
    # or a local Hydra config path. The HF builder resolves the id to a config + ckpt;
    # plain build_sam2_video_predictor expects the bare config name.
    if "/" in sam2_model and not sam2_model.endswith(".yaml"):
        predictor = build_sam2_video_predictor_hf(sam2_model, device=device)
    else:
        predictor = build_sam2_video_predictor(sam2_model, device=device)

    text_input = prompt.strip() or "the liquid inside the bottle"
    box_xyxy = _grounding_dino_box_xyxy(frames[0], text_input, device=device)

    state = predictor.init_state(video_path=video_path)
    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=0,
        box=box_xyxy,
    )
    target_frames = int(frames.shape[0])
    masks: list[np.ndarray] = [np.zeros(frames.shape[1:3], dtype=bool) for _ in range(target_frames)]
    # SAM2 propagates over the full video that init_state opened; cap at the
    # frame budget we asked _read_video_frames_uint8 to read so max_pairs is
    # actually respected (otherwise the pre-sized masks list would IndexError).
    for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
        if frame_idx >= target_frames:
            break
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
    # Hash the mask volume itself: two mask runs with the same sam2_model/prompt
    # strings can still produce different masks (model patch, video re-encode,
    # nondeterminism). Without this, PIV would silently serve stale results.
    mask_bytes_hash = hashlib.sha256(np.ascontiguousarray(mask.masks).tobytes()).hexdigest()[:16]
    cfg = _config_hash(
        {
            "window_size": window_size,
            "overlap": overlap,
            "fps": fps,
            "max_frames": max_frames,
            "sam2_model": mask.sam2_model,
            "prompt": mask.prompt,
            "mask_hash": mask_bytes_hash,
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


def get_bottle_centroid_trace(
    *,
    video_path: str,
    sam2_model: str,
    bottle_prompt: str,
    fps: float,
    max_pairs: int | None,
    cache_dir: str | Path,
) -> BottleTrace:
    """Per-frame bottle centroid (x, y) and bbox, derived from a SAM2 pass.

    Reuses get_liquid_mask_video with the bottle prompt rather than the liquid
    prompt; the cache key includes the prompt so this gets its own cache file
    independent of the liquid mask. Frames where SAM2 returns an empty mask
    are marked invalid and forward-filled from the previous valid frame so
    consumers can take time-derivatives without nan-spikes.
    """
    bottle_mask = get_liquid_mask_video(
        video_path=video_path,
        sam2_model=sam2_model,
        prompt=bottle_prompt,
        fps=fps,
        max_pairs=max_pairs,
        cache_dir=cache_dir,
    )
    masks = bottle_mask.masks  # (T, H, W) bool
    T, H, W = masks.shape
    centroids = np.full((T, 2), np.nan, dtype=np.float64)
    bboxes = np.full((T, 4), np.nan, dtype=np.float64)
    valid = np.zeros(T, dtype=bool)

    ys = np.arange(H, dtype=np.float64)[:, None]
    xs = np.arange(W, dtype=np.float64)[None, :]

    for t in range(T):
        m = masks[t]
        n = float(m.sum())
        if n <= 0:
            continue
        cx = float((m.astype(np.float64) * xs).sum() / n)
        cy = float((m.astype(np.float64) * ys).sum() / n)
        rows = np.where(m.any(axis=1))[0]
        cols = np.where(m.any(axis=0))[0]
        x0, x1 = float(cols.min()), float(cols.max())
        y0, y1 = float(rows.min()), float(rows.max())
        centroids[t] = (cx, cy)
        bboxes[t] = (x0, y0, x1, y1)
        valid[t] = True

    last_good = -1
    for t in range(T):
        if valid[t]:
            last_good = t
        elif last_good >= 0:
            centroids[t] = centroids[last_good]
            bboxes[t] = bboxes[last_good]
    if not valid.any():
        raise RuntimeError(
            f"get_bottle_centroid_trace: SAM2 returned empty masks for every frame "
            f"of {video_path} (prompt={bottle_prompt!r})"
        )
    first_good = int(np.argmax(valid))
    if first_good > 0:
        centroids[:first_good] = centroids[first_good]
        bboxes[:first_good] = bboxes[first_good]

    return BottleTrace(centroid_xy=centroids, bbox_xyxy=bboxes, valid=valid)


def get_raft_flow_field(
    *,
    video_path: str,
    mask: LiquidMaskResult,
    window_size: int,
    overlap: int,
    fps: float,
    cache_dir: str | Path,
    max_frames: int | None = None,
    min_mask_frac: float = 0.5,
) -> PIVField:
    """Pooled torchvision-RAFT optical flow on a PIV-style grid.

    OpenPIV's cross-correlation rejects ~99% of cells inside opaque dark
    cola (the autocorrelation peak doesn't dominate the noise floor), so
    the velocity metric is unusable on this scene. RAFT is a learned dense
    estimator with global context that works on uniformly-coloured regions,
    bringing per-cell validity from ~1% up to the liquid-mask fraction.

    To stay shape-compatible with the existing PIV pipeline (so EPE /
    cosine / divergence run unchanged) we pool the dense flow into the
    same (Hp, Wp) grid OpenPIV would have produced for the given
    window/overlap settings, masking each cell to liquid pixels only.
    A cell is `valid` when ≥10% of its window-area is liquid in BOTH the
    pre and post frames.
    """
    import cv2

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mask_bytes_hash = hashlib.sha256(np.ascontiguousarray(mask.masks).tobytes()).hexdigest()[:16]
    cfg = _config_hash(
        {
            "estimator": "raft_pooled_v1",
            "window_size": window_size,
            "overlap": overlap,
            "fps": fps,
            "max_frames": max_frames,
            "sam2_model": mask.sam2_model,
            "prompt": mask.prompt,
            "mask_hash": mask_bytes_hash,
            "min_mask_frac": float(min_mask_frac),
        }
    )
    cache_file = _cache_path(cache_dir, video_path, "raft_pooled", cfg)
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    import torch

    from .flow_utils import flow_raft, load_raft_model

    rgb = _read_video_frames_uint8(video_path, max_frames=max_frames)
    if rgb.shape[0] < 2:
        raise RuntimeError(f"get_raft_flow_field: need ≥2 frames, got {rgb.shape[0]}")
    masks = mask.masks
    if masks.shape[0] < rgb.shape[0]:
        raise ValueError(
            f"get_raft_flow_field: mask has {masks.shape[0]} frames, video has {rgb.shape[0]}"
        )

    H, W = rgb.shape[1], rgb.shape[2]
    step = max(1, int(window_size) - int(overlap))
    half = int(window_size) // 2
    ys = np.arange(half, H - half + 1, step, dtype=int)
    xs = np.arange(half, W - half + 1, step, dtype=int)
    if ys.size == 0 or xs.size == 0:
        raise RuntimeError(
            f"get_raft_flow_field: window/overlap too large for video size {H}x{W}"
        )
    Hp, Wp = ys.size, xs.size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = load_raft_model(device)

    T = rgb.shape[0]
    us = np.full((T - 1, Hp, Wp), np.nan, dtype=np.float32)
    vs = np.full((T - 1, Hp, Wp), np.nan, dtype=np.float32)
    valids = np.zeros((T - 1, Hp, Wp), dtype=bool)

    ks = (int(window_size), int(window_size))
    # Default 0.5 (formerly 0.1) — lower fractions let boundary cells leak in,
    # and on partially-transparent renders those cells pick up apparent
    # background motion through the glass instead of liquid motion.
    frac = max(0.0, min(1.0, float(min_mask_frac)))
    min_pixels = max(4.0, frac * float(window_size) ** 2)

    for t in range(T - 1):
        prev_bgr = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR)
        next_bgr = cv2.cvtColor(rgb[t + 1], cv2.COLOR_RGB2BGR)
        flow = flow_raft(model, transform, prev_bgr, next_bgr, device)  # (H, W, 2)

        joint_mask = (masks[t] & masks[t + 1]).astype(np.float32)
        # Box-pool the masked flow then divide by the per-cell pixel count
        # — much faster than the naive per-cell loop and numerically equivalent.
        u_masked = flow[..., 0] * joint_mask
        v_masked = flow[..., 1] * joint_mask
        count = cv2.boxFilter(joint_mask, ddepth=-1, ksize=ks, normalize=False)
        u_sum = cv2.boxFilter(u_masked, ddepth=-1, ksize=ks, normalize=False)
        v_sum = cv2.boxFilter(v_masked, ddepth=-1, ksize=ks, normalize=False)

        cell_count = count[ys[:, None], xs[None, :]]
        cell_u = u_sum[ys[:, None], xs[None, :]]
        cell_v = v_sum[ys[:, None], xs[None, :]]
        cell_valid = cell_count >= min_pixels
        with np.errstate(invalid="ignore", divide="ignore"):
            u_avg = np.where(cell_valid, cell_u / np.maximum(cell_count, 1.0), np.nan)
            v_avg = np.where(cell_valid, cell_v / np.maximum(cell_count, 1.0), np.nan)
        us[t] = u_avg.astype(np.float32)
        vs[t] = v_avg.astype(np.float32)
        valids[t] = cell_valid

    field = PIVField(u=us, v=vs, valid=valids)
    with open(cache_file, "wb") as f:
        pickle.dump(field, f)
    return field


def resize_piv_to_common_grid(unity: PIVField, wm: PIVField) -> tuple[PIVField, PIVField]:
    """Resize both PIV fields to a common (H, W) grid via nearest-neighbour
    spatial sampling so EPE / cosine / divergence comparisons are well-defined
    when the source videos rendered at different resolutions.

    We use the smaller (H, W) of the two as the target so we never invent
    samples. Each output cell gets the source cell whose center is closest;
    `valid` is propagated the same way (a downsampled cell is invalid if
    its source cell was invalid). Per-frame velocity magnitudes are NOT
    rescaled here — callers should normalize px/sec against bottle width
    before any cross-resolution physical comparison.
    """
    if unity.u.shape == wm.u.shape:
        return unity, wm
    if unity.u.ndim != 3 or wm.u.ndim != 3:
        raise ValueError(
            f"resize_piv_to_common_grid: expected 3-D fields, got {unity.u.shape} and {wm.u.shape}"
        )
    Tu, Hu, Wu = unity.u.shape
    Tw, Hw, Ww = wm.u.shape
    if Tu != Tw:
        n = min(Tu, Tw)
        unity = PIVField(u=unity.u[:n], v=unity.v[:n], valid=unity.valid[:n])
        wm = PIVField(u=wm.u[:n], v=wm.v[:n], valid=wm.valid[:n])
    H_target = min(Hu, Hw)
    W_target = min(Wu, Ww)

    def _resample(field: PIVField) -> PIVField:
        T, H, W = field.u.shape
        if (H, W) == (H_target, W_target):
            return field
        ys = np.linspace(0, H - 1, H_target).round().astype(int)
        xs = np.linspace(0, W - 1, W_target).round().astype(int)
        u = field.u[:, ys[:, None], xs[None, :]]
        v = field.v[:, ys[:, None], xs[None, :]]
        valid = field.valid[:, ys[:, None], xs[None, :]]
        return PIVField(u=u, v=v, valid=valid)

    return _resample(unity), _resample(wm)
