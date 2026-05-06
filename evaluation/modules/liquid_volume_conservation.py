# WMEval/evaluation/modules/liquid_volume_conservation.py
"""Mask-area conservation metrics for sealed-bottle sloshing.

For a sealed container under a fixed camera (c2_static_side regime) the
projected liquid mask area must stay constant up to anti-aliasing noise.
Two failure modes matter:

  - Short-horizon jitter (mask flickers frame-to-frame) — captured by
    `mask_area_std_normalized` (per-frame std / mean).
  - Long-horizon leakage / spurious growth — captured by
    `mask_area_drift_per_sec` (linear-fit slope / mean, in 1/sec).

Each is reported separately for Unity, WM, and as `delta = wm − unity`
(signed: positive means WM jitters or drifts more than Unity).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..context import EvaluationContext
from . import _liquid_common as _lc


def _normalized_std(area: np.ndarray) -> float:
    mean = float(area.mean())
    if mean <= 1e-9:
        return float("nan")
    return float(area.std() / mean)


def _drift_per_sec(area: np.ndarray, fps: float) -> float:
    mean = float(area.mean())
    if mean <= 1e-9 or area.size < 2:
        return float("nan")
    t = np.arange(area.size, dtype=np.float64) / float(fps)
    slope, _ = np.polyfit(t, area, 1)
    return float(slope) / mean


def compute_volume_conservation_from_masks(
    *,
    unity_masks: np.ndarray,
    wm_masks: np.ndarray,
    fps: float,
) -> Dict[str, Any]:
    if fps <= 0:
        raise ValueError("compute_volume_conservation_from_masks: fps must be positive")

    unity_area = _lc._mask_areas(unity_masks)
    wm_area = _lc._mask_areas(wm_masks)

    std_u = _normalized_std(unity_area)
    std_w = _normalized_std(wm_area)
    drift_u = _drift_per_sec(unity_area, fps)
    drift_w = _drift_per_sec(wm_area, fps)

    def _delta(unity: float, wm: float) -> float:
        if not (np.isfinite(unity) and np.isfinite(wm)):
            return float("nan")
        return float(wm - unity)

    return {
        "mask_area_std_normalized": {
            "unity": std_u,
            "wm": std_w,
            "delta": _delta(std_u, std_w),
        },
        "mask_area_drift_per_sec": {
            "unity": drift_u,
            "wm": drift_w,
            "delta": _delta(drift_u, drift_w),
        },
    }


def analyze_liquid_volume_conservation_from_context(
    *,
    context: EvaluationContext,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    del device
    arts = context.artifacts

    if not context.reference_video or not context.candidate_video:
        return {
            "module": "liquid_volume_conservation",
            "error": "both reference_video and candidate_video required",
        }

    cache_root = Path(arts.get("liquid_cache_root", "runs/liquid_cache"))
    sam2_model = str(arts.get("liquid_sam2_model", "facebook/sam2.1-hiera-small"))
    sam2_prompt = str(arts.get("liquid_sam2_prompt", "the liquid inside the bottle"))
    fps = _lc.require_matched_fps(context, module_name="liquid_volume_conservation")
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
    metrics = compute_volume_conservation_from_masks(
        unity_masks=unity_mask.masks[:n],
        wm_masks=wm_mask.masks[:n],
        fps=fps,
    )
    return {
        "module": "liquid_volume_conservation",
        **metrics,
        "diagnostics": {"frames_evaluated": int(n)},
    }
