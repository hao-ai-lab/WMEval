# WMEval/evaluation/modules/slosh_spectrum.py
"""Slosh-frequency spectrum metric for sealed-bottle sloshing.

We trace the vertical centroid of the per-frame liquid mask, take the
real FFT, drop the DC bin, and report:

  - The dominant frequency (Hz) for Unity and WM, and their ratio.
  - The L1 distance between unit-area normalized power spectra
    (catches mode-shape mismatch even when the peak frequency is close).

A static-camera, sloshing bottle has a clear peak near the cavity's
natural frequency; a model that fails to capture sloshing produces a
flat or noisy spectrum.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..context import EvaluationContext
from . import _liquid_common as _lc


_MIN_FRAMES = 8


def _centroid_y_trace(masks: np.ndarray) -> np.ndarray:
    """Return shape (T,) vertical centroid; NaN where the mask is empty."""
    if masks.ndim != 3:
        raise ValueError(f"_centroid_y_trace: expected (T,H,W) got {masks.shape}")
    T, H, _ = masks.shape
    y_idx = np.arange(H, dtype=np.float64)[:, None]  # (H, 1)
    out = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        m = masks[t]
        n = float(m.sum())
        if n <= 0:
            continue
        out[t] = float((m.astype(np.float64) * y_idx).sum() / n)
    return out


def _power_spectrum(trace: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs_hz, power) excluding the DC bin.

    The trace is mean-centered before FFT to suppress the DC component.
    Returns empty arrays when the trace has fewer than _MIN_FRAMES finite samples.
    """
    finite = np.isfinite(trace)
    if int(finite.sum()) < _MIN_FRAMES:
        return np.array([]), np.array([])
    x = trace.copy()
    x[~finite] = float(np.nanmean(x))
    x = x - x.mean()
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(fps))
    # Drop DC bin (index 0).
    return freqs[1:], spectrum[1:]


def _dominant_frequency(freqs: np.ndarray, power: np.ndarray) -> float:
    if freqs.size == 0 or power.size == 0:
        return float("nan")
    return float(freqs[int(np.argmax(power))])


def _normalized(power: np.ndarray) -> np.ndarray:
    total = power.sum()
    if total <= 1e-12:
        # Honour the "unit-area normalized" contract: a near-flat spectrum
        # contributes zero mass instead of un-normalized noise to spectral_l1.
        return np.zeros_like(power)
    return power / total


def compute_slosh_spectrum_from_masks(
    *,
    unity_masks: np.ndarray,
    wm_masks: np.ndarray,
    fps: float,
) -> Dict[str, Any]:
    if fps <= 0:
        raise ValueError("compute_slosh_spectrum_from_masks: fps must be positive")

    unity_trace = _centroid_y_trace(unity_masks)
    wm_trace = _centroid_y_trace(wm_masks)

    freqs_u, power_u = _power_spectrum(unity_trace, fps)
    freqs_w, power_w = _power_spectrum(wm_trace, fps)

    f_u = _dominant_frequency(freqs_u, power_u)
    f_w = _dominant_frequency(freqs_w, power_w)
    if np.isfinite(f_u) and np.isfinite(f_w) and f_u > 1e-6:
        ratio = float(f_w / f_u)
    else:
        ratio = float("nan")
    # Signed delta: positive = WM peak is higher than Unity's; negative = lower.
    delta_hz = float(f_w - f_u) if (np.isfinite(f_u) and np.isfinite(f_w)) else float("nan")

    if power_u.size > 0 and power_w.size > 0 and power_u.size == power_w.size:
        l1 = float(np.abs(_normalized(power_u) - _normalized(power_w)).sum())
    else:
        l1 = float("nan")

    return {
        "dominant_frequency_hz": {"unity": f_u, "wm": f_w, "ratio": ratio, "delta_hz": delta_hz},
        "spectral_l1": l1,
    }


def analyze_slosh_spectrum_from_context(
    *,
    context: EvaluationContext,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    del device
    arts = context.artifacts

    if not context.reference_video or not context.candidate_video:
        return {
            "module": "slosh_spectrum",
            "error": "both reference_video and candidate_video required",
        }

    cache_root = Path(arts.get("liquid_cache_root", "runs/liquid_cache"))
    sam2_model = str(arts.get("liquid_sam2_model", "facebook/sam2.1-hiera-small"))
    sam2_prompt = str(arts.get("liquid_sam2_prompt", "the liquid inside the bottle"))
    fps = _lc.require_matched_fps(context, module_name="slosh_spectrum")
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
    metrics = compute_slosh_spectrum_from_masks(
        unity_masks=unity_mask.masks[:n],
        wm_masks=wm_mask.masks[:n],
        fps=fps,
    )
    return {
        "module": "slosh_spectrum",
        **metrics,
        "diagnostics": {"frames_evaluated": int(n)},
    }
