# WMEval/evaluation/modules/liquid_surface_orientation.py
"""Free-surface orientation metric for sealed-bottle sloshing.

For each frame we extract the topmost True pixel per column from the
liquid mask — this traces the air/liquid interface in image space.
A linear fit yields the surface tilt (deg from horizontal). The
per-sequence metric is the RMS of |tilt| across frames; for a static-
camera, upright bottle this should be ~0.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..context import EvaluationContext
from . import _liquid_common as _lc


def _per_frame_tilt_deg(masks: np.ndarray) -> np.ndarray:
    """Return shape (T,) tilt in degrees from horizontal for each frame.

    NaN for frames where the surface is not extractable (no liquid columns,
    or only one column).
    """
    if masks.ndim != 3:
        raise ValueError(f"_per_frame_tilt_deg: expected (T,H,W) got {masks.shape}")
    T, H, W = masks.shape
    tilts = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        m = masks[t]
        col_has_liquid = m.any(axis=0)
        if int(col_has_liquid.sum()) < 2:
            continue
        # Topmost True row per column (small index = high in image).
        # argmax returns 0 for all-False columns; we filter those out.
        top_rows = np.argmax(m, axis=0)
        xs = np.where(col_has_liquid)[0].astype(np.float64)
        ys = top_rows[col_has_liquid].astype(np.float64)
        if xs.size < 2:
            continue
        slope, _ = np.polyfit(xs, ys, 1)
        tilts[t] = float(np.degrees(np.arctan(slope)))
    return tilts


def _rms_abs(tilts: np.ndarray) -> float:
    finite = tilts[np.isfinite(tilts)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite ** 2)))


def compute_surface_orientation_from_masks(
    *,
    unity_masks: np.ndarray,
    wm_masks: np.ndarray,
) -> Dict[str, Any]:
    unity_tilts = _per_frame_tilt_deg(unity_masks)
    wm_tilts = _per_frame_tilt_deg(wm_masks)
    rms_u = _rms_abs(unity_tilts)
    rms_w = _rms_abs(wm_tilts)
    # Signed: delta = wm − unity, so positive = WM tilts more than Unity.
    if np.isfinite(rms_u) and np.isfinite(rms_w):
        delta = float(rms_w - rms_u)
    else:
        delta = float("nan")
    return {
        "surface_tilt_residual_deg": {
            "unity": rms_u,
            "wm": rms_w,
            "delta": delta,
        }
    }


def analyze_liquid_surface_orientation_from_context(
    *,
    context: EvaluationContext,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    del device
    arts = context.artifacts

    if not context.reference_video or not context.candidate_video:
        return {
            "module": "liquid_surface_orientation",
            "error": "both reference_video and candidate_video required",
        }

    cache_root = Path(arts.get("liquid_cache_root", "runs/liquid_cache"))
    sam2_model = str(arts.get("liquid_sam2_model", "facebook/sam2.1-hiera-small"))
    sam2_prompt = str(arts.get("liquid_sam2_prompt", "the liquid inside the bottle"))
    fps = _lc.require_matched_fps(context, module_name="liquid_surface_orientation")
    max_pairs = int(arts.get("liquid_max_pairs", 0)) or None

    unity_mask = _lc.get_liquid_mask_video(
        video_path=context.reference_video,
        sam2_model=sam2_model,
        prompt=sam2_prompt,
        fps=fps,
        max_pairs=max_pairs,
        cache_dir=cache_root,
    )
    wm_mask = _lc.get_liquid_mask_video(
        video_path=context.candidate_video,
        sam2_model=sam2_model,
        prompt=sam2_prompt,
        fps=fps,
        max_pairs=max_pairs,
        cache_dir=cache_root,
    )

    n = min(unity_mask.masks.shape[0], wm_mask.masks.shape[0])
    metrics = compute_surface_orientation_from_masks(
        unity_masks=unity_mask.masks[:n],
        wm_masks=wm_mask.masks[:n],
    )
    return {
        "module": "liquid_surface_orientation",
        **metrics,
        "diagnostics": {"frames_evaluated": int(n)},
    }
