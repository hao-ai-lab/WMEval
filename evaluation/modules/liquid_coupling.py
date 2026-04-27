"""Liquid-coupling metric: lagged Pearson correlation between bottle tilt and pour rate."""
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


def compute_lagged_correlation(
    *,
    tilt: np.ndarray,
    pour_rate: np.ndarray,
    fps: float,
    max_lag_sec: float = 1.0,
) -> Dict[str, float]:
    """Find τ ∈ [0, max_lag_sec] maximizing corr(tilt[t], pour_rate[t+τ]).

    Returns the maximum correlation and the lag (sec) at which it occurs. Defaults
    to lag=0 if all candidates are NaN.
    """
    n = min(len(tilt), len(pour_rate))
    if n < 3:
        return {"correlation": float("nan"), "lag_sec": 0.0}
    max_lag_frames = max(1, int(round(max_lag_sec * fps)))
    best_corr = float("-inf")
    best_lag = 0
    for lag in range(0, max_lag_frames + 1):
        a = tilt[: n - lag]
        b = pour_rate[lag:n]
        c = _pearsonr_safe(a, b)
        if np.isfinite(c) and c > best_corr:
            best_corr = c
            best_lag = lag
    if not np.isfinite(best_corr):
        return {"correlation": float("nan"), "lag_sec": 0.0}
    return {"correlation": float(best_corr), "lag_sec": float(best_lag) / float(fps)}


def analyze_liquid_coupling_from_context(
    *,
    context: EvaluationContext,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    del device
    arts = context.artifacts

    log_path = arts.get("liquid_state_unity_log_path")
    if not log_path:
        return {"module": "liquid_coupling", "error": "missing artifacts: ['liquid_state_unity_log_path']"}
    if not context.reference_video or not context.candidate_video:
        return {"module": "liquid_coupling", "error": "both reference_video and candidate_video required"}

    cache_root = Path(arts.get("liquid_cache_root", "runs/liquid_cache"))
    sam2_model = str(arts.get("liquid_sam2_model", "facebook/sam2.1-hiera-small"))
    sam2_prompt = str(arts.get("liquid_sam2_prompt", "the liquid inside the bottle"))
    fps = float(context.reference.metadata.fps)
    max_pairs = int(arts.get("liquid_max_pairs", 0)) or None
    bottle_volume_ml = float(arts.get("liquid_bottle_local_volume_ml", 500.0))
    max_lag_sec = float(arts.get("liquid_coupling_max_lag_sec", 1.0))

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
    if n < 3:
        return {"module": "liquid_coupling", "error": f"too few frames: n={n}"}

    calibration = _lc.fit_fill_calibration(
        unity_masks=unity_mask.masks[:n], unity_log_fill=log.fill_fraction[:n],
    )
    wm_fill = _lc.estimate_fill_curve_from_mask(wm_mask.masks[:n], calibration=calibration)
    # Negative values (back-flow) are kept — they are a model-failure signal, not noise.
    wm_pour_rate = np.zeros(n, dtype=np.float32)
    wm_pour_rate[1:] = -(wm_fill[1:] - wm_fill[:-1]) * fps * bottle_volume_ml

    tilt = log.tilt_deg[:n]
    unity_pour_rate = log.pour_rate[:n]

    coupling_unity = compute_lagged_correlation(
        tilt=tilt, pour_rate=unity_pour_rate, fps=fps, max_lag_sec=max_lag_sec,
    )
    coupling_wm = compute_lagged_correlation(
        tilt=tilt, pour_rate=wm_pour_rate, fps=fps, max_lag_sec=max_lag_sec,
    )

    delta = float(coupling_unity["correlation"] - coupling_wm["correlation"]) if (
        np.isfinite(coupling_unity["correlation"]) and np.isfinite(coupling_wm["correlation"])
    ) else float("nan")

    return {
        "module": "liquid_coupling",
        "tilt_pour_correlation": {
            "unity": coupling_unity["correlation"],
            "wm": coupling_wm["correlation"],
            "delta": delta,
            "lag_unity_sec": coupling_unity["lag_sec"],
            "lag_wm_sec": coupling_wm["lag_sec"],
        },
        "diagnostics": {"frames_evaluated": int(n)},
    }
