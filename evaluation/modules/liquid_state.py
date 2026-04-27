# WMEval/evaluation/modules/liquid_state.py
"""Liquid-state metrics: fill curve, pour onset/rate, spill, surface height.

This module loads the Unity ground-truth log (authoritative scalar timeseries),
estimates the same quantities from the WM video via SAM2 + a fitted
calibration, then reports per-axis metrics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..context import EvaluationContext
from . import _liquid_common as _lc


def _pearsonr_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size < 3:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _normalized_dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized DTW distance: fastdtw L1 / (max(len(a), len(b)) * max(|a|, |b|, eps))."""
    from fastdtw import fastdtw

    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    n = max(aa.size, bb.size)
    scale = max(float(np.max(np.abs(aa))), float(np.max(np.abs(bb))), 1e-6)
    distance, _path = fastdtw(aa, bb, dist=lambda x, y: float(abs(x - y)))
    return float(distance) / float(n * scale)


def compute_state_metrics_from_arrays(
    *,
    unity_fill: np.ndarray,
    wm_fill: np.ndarray,
    fps: float,
    unity_pour_rate: np.ndarray,
    wm_pour_rate: np.ndarray,
    unity_surface_height: np.ndarray,
    wm_surface_height: np.ndarray,
    unity_spilled_final_ml: float,
    wm_spilled_estimate_ml: float,
    onset_threshold_drop: float = 0.005,
    onset_k_consecutive: int = 5,
    onset_smoothing_window: int = 3,
) -> Dict[str, Any]:
    """Compute the §7.2 state metrics from already-aligned arrays.

    Pure function — no I/O — exposed for unit-testing.
    """
    unity_onset = _lc.detect_pour_onset_from_curve(
        unity_fill, fps,
        threshold_drop=onset_threshold_drop, k_consecutive=onset_k_consecutive,
        smoothing_window=onset_smoothing_window,
    )
    wm_onset = _lc.detect_pour_onset_from_curve(
        wm_fill, fps,
        threshold_drop=onset_threshold_drop, k_consecutive=onset_k_consecutive,
        smoothing_window=onset_smoothing_window,
    )
    if unity_onset is None or wm_onset is None:
        onset_offset = float("nan")
    else:
        onset_offset = float(wm_onset - unity_onset)

    return {
        "fill": {
            "curve_correlation": _pearsonr_safe(unity_fill, wm_fill),
            "mae": float(np.mean(np.abs(np.asarray(unity_fill) - np.asarray(wm_fill)))),
        },
        "pour": {
            "onset_offset_sec": onset_offset,
            "rate_curve_dtw": _normalized_dtw(unity_pour_rate, wm_pour_rate),
            "unity_onset_sec": unity_onset,
            "wm_onset_sec": wm_onset,
        },
        "spill": {
            "final_volume_error_ml": float(abs(wm_spilled_estimate_ml - unity_spilled_final_ml)),
            "unity_final_ml": float(unity_spilled_final_ml),
            "wm_estimate_ml": float(wm_spilled_estimate_ml),
        },
        "surface": {
            "height_trace_correlation": _pearsonr_safe(unity_surface_height, wm_surface_height),
        },
    }


def _topmost_mask_y_per_frame(masks: np.ndarray) -> np.ndarray:
    """For each frame, return the row index of the topmost mask pixel (or NaN)."""
    out = np.full(masks.shape[0], np.nan, dtype=np.float32)
    for t in range(masks.shape[0]):
        rows = np.where(masks[t].any(axis=1))[0]
        if rows.size > 0:
            out[t] = float(rows.min())
    return out


def analyze_liquid_state_from_context(
    *,
    context: EvaluationContext,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    del device
    arts = context.artifacts

    if not context.reference_video or not context.candidate_video:
        return {"module": "liquid_state", "error": "both reference_video and candidate_video required"}
    log_path = arts.get("liquid_state_unity_log_path")
    if not log_path:
        return {"module": "liquid_state", "error": "missing artifacts: ['liquid_state_unity_log_path']"}

    cache_root = Path(arts.get("liquid_cache_root", "runs/liquid_cache"))
    sam2_model = str(arts.get("liquid_sam2_model", "facebook/sam2.1-hiera-small"))
    sam2_prompt = str(arts.get("liquid_sam2_prompt", "the liquid inside the bottle"))
    fps = float(context.reference.metadata.fps)
    max_pairs = int(arts.get("liquid_max_pairs", 0)) or None
    bottle_volume_ml = float(arts.get("liquid_bottle_local_volume_ml", 500.0))

    log = _lc.load_unity_log(log_path)
    unity_mask = _lc.get_liquid_mask_video(
        video_path=context.reference_video, sam2_model=sam2_model, prompt=sam2_prompt,
        fps=fps, max_pairs=max_pairs, cache_dir=cache_root,
    )
    wm_mask = _lc.get_liquid_mask_video(
        video_path=context.candidate_video, sam2_model=sam2_model, prompt=sam2_prompt,
        fps=fps, max_pairs=max_pairs, cache_dir=cache_root,
    )

    n = min(unity_mask.num_frames, wm_mask.num_frames, log.num_frames)
    if n < 2:
        return {"module": "liquid_state", "error": f"too few aligned frames: n={n}"}

    calibration = _lc.fit_fill_calibration(
        unity_masks=unity_mask.masks[:n], unity_log_fill=log.fill_fraction[:n],
    )
    wm_fill = _lc.estimate_fill_curve_from_mask(wm_mask.masks[:n], calibration=calibration)
    unity_fill = log.fill_fraction[:n]

    # WM pour-rate is finite-differenced from wm_fill (no Unity log on the WM side).
    # Negative values (back-flow) are kept — they are a model-failure signal, not noise.
    wm_pour_rate = np.zeros(n, dtype=np.float32)
    wm_pour_rate[1:] = -(wm_fill[1:] - wm_fill[:-1]) * fps * bottle_volume_ml
    unity_pour_rate = log.pour_rate[:n]

    # Surface-height proxy on WM side: topmost mask pixel y, in normalized [0,1] image units.
    wm_top_y = _topmost_mask_y_per_frame(wm_mask.masks[:n]) / float(wm_mask.masks.shape[1])
    unity_surface = log.surface_height[:n]

    wm_spill_estimate = float((log.initial_fill_fraction - float(wm_fill[-1])) * bottle_volume_ml)
    wm_spill_estimate = max(0.0, wm_spill_estimate)

    metrics = compute_state_metrics_from_arrays(
        unity_fill=unity_fill, wm_fill=wm_fill, fps=fps,
        unity_pour_rate=unity_pour_rate, wm_pour_rate=wm_pour_rate,
        unity_surface_height=unity_surface, wm_surface_height=wm_top_y,
        unity_spilled_final_ml=float(log.spilled_volume[n - 1]),
        wm_spilled_estimate_ml=wm_spill_estimate,
    )

    return {
        "module": "liquid_state",
        "fill": metrics["fill"],
        "pour": metrics["pour"],
        "spill": metrics["spill"],
        "surface": metrics["surface"],
        "diagnostics": {
            "fill_calibration_r2": calibration.r2,
            "frames_evaluated": int(n),
        },
    }
