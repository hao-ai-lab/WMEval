# WMEval/evaluation/modules/liquid_velocity.py
"""Liquid-velocity metrics: EPE, cosine, divergence/curl residuals, temporal consistency."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..context import EvaluationContext
from . import _liquid_common as _lc


def _epe_stats(diff_norm: np.ndarray) -> Dict[str, float]:
    flat = diff_norm[np.isfinite(diff_norm)]
    if flat.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "p90": float(np.percentile(flat, 90)),
    }


def _cosine_similarity(u_a, v_a, u_b, v_b, joint_valid) -> float:
    # Compute in float64 so eps is not absorbed below the float32 ULP (~1.19e-7 at unit magnitudes).
    u_a64 = np.asarray(u_a, dtype=np.float64)
    v_a64 = np.asarray(v_a, dtype=np.float64)
    u_b64 = np.asarray(u_b, dtype=np.float64)
    v_b64 = np.asarray(v_b, dtype=np.float64)
    eps = 1e-12
    dot = u_a64 * u_b64 + v_a64 * v_b64
    na = np.sqrt(u_a64 * u_a64 + v_a64 * v_a64) + eps
    nb = np.sqrt(u_b64 * u_b64 + v_b64 * v_b64) + eps
    cos = dot / (na * nb)
    cos = cos[joint_valid & np.isfinite(cos)]
    if cos.size == 0:
        return float("nan")
    return float(np.mean(cos))


def _temporal_consistency(field: _lc.PIVField) -> float:
    if field.u.shape[0] < 2:
        return float("nan")
    du = field.u[1:] - field.u[:-1]
    dv = field.v[1:] - field.v[:-1]
    norm = np.sqrt(du * du + dv * dv)
    valid_pair = field.valid[1:] & field.valid[:-1]
    norm = norm[valid_pair & np.isfinite(norm)]
    if norm.size == 0:
        return float("nan")
    return float(np.mean(norm))


def compute_velocity_metrics_from_fields(
    *,
    unity: _lc.PIVField,
    wm: _lc.PIVField,
) -> Dict[str, Any]:
    if unity.u.shape != wm.u.shape:
        n = min(unity.u.shape[0], wm.u.shape[0])
        unity = _lc.PIVField(u=unity.u[:n], v=unity.v[:n], valid=unity.valid[:n])
        wm = _lc.PIVField(u=wm.u[:n], v=wm.v[:n], valid=wm.valid[:n])

    joint_valid = unity.valid & wm.valid
    du = wm.u - unity.u
    dv = wm.v - unity.v
    diff_norm = np.where(joint_valid, np.sqrt(du * du + dv * dv), np.nan)

    div_u, curl_u = _lc.divergence_curl_2d(unity)
    div_w, curl_w = _lc.divergence_curl_2d(wm)
    div_residual = np.abs(div_w - div_u)
    div_residual = div_residual[np.isfinite(div_residual)]
    div_residual_mean = float(np.mean(div_residual)) if div_residual.size > 0 else float("nan")

    curl_residual = np.abs(curl_w - curl_u)
    curl_residual = curl_residual[np.isfinite(curl_residual)]
    curl_mean_err = float(np.mean(curl_residual)) if curl_residual.size > 0 else float("nan")
    curl_std_err = float(np.std(curl_residual)) if curl_residual.size > 0 else float("nan")

    tc_unity = _temporal_consistency(unity)
    tc_wm = _temporal_consistency(wm)
    if np.isfinite(tc_unity) and tc_unity > 1e-8 and np.isfinite(tc_wm):
        tc_ratio = float(tc_wm / tc_unity)
    else:
        tc_ratio = float("nan")

    return {
        "epe": _epe_stats(diff_norm),
        "cosine_similarity": _cosine_similarity(unity.u, unity.v, wm.u, wm.v, joint_valid),
        "divergence_residual": div_residual_mean,
        "curl_stats": {"mean_error": curl_mean_err, "std_error": curl_std_err},
        "temporal_consistency": tc_ratio,
    }


def analyze_liquid_velocity_from_context(
    *,
    context: EvaluationContext,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    del device
    arts = context.artifacts

    if not context.reference_video or not context.candidate_video:
        return {"module": "liquid_velocity", "error": "both reference_video and candidate_video required"}

    cache_root = Path(arts.get("liquid_cache_root", "runs/liquid_cache"))
    sam2_model = str(arts.get("liquid_sam2_model", "facebook/sam2.1-hiera-small"))
    sam2_prompt = str(arts.get("liquid_sam2_prompt", "the liquid inside the bottle"))
    window_size = int(arts.get("liquid_piv_window_size", 32))
    overlap = int(arts.get("liquid_piv_overlap", 16))
    fps = _lc.require_matched_fps(context, module_name="liquid_velocity")
    max_pairs = int(arts.get("liquid_max_pairs", 0)) or None

    unity_mask = _lc.get_liquid_mask_video(
        video_path=context.reference_video, sam2_model=sam2_model, prompt=sam2_prompt,
        fps=fps, max_pairs=max_pairs, cache_dir=cache_root,
    )
    wm_mask = _lc.get_liquid_mask_video(
        video_path=context.candidate_video, sam2_model=sam2_model, prompt=sam2_prompt,
        fps=fps, max_pairs=max_pairs, cache_dir=cache_root,
    )
    unity_field = _lc.get_piv_field(
        video_path=context.reference_video, mask=unity_mask,
        window_size=window_size, overlap=overlap, fps=fps,
        cache_dir=cache_root, max_frames=max_pairs,
    )
    wm_field = _lc.get_piv_field(
        video_path=context.candidate_video, mask=wm_mask,
        window_size=window_size, overlap=overlap, fps=fps,
        cache_dir=cache_root, max_frames=max_pairs,
    )

    metrics = compute_velocity_metrics_from_fields(unity=unity_field, wm=wm_field)

    unity_valid_frac = float(np.mean(unity_field.valid))
    wm_valid_frac = float(np.mean(wm_field.valid))

    return {
        "module": "liquid_velocity",
        **metrics,
        "diagnostics": {
            "piv_validity_unity": unity_valid_frac,
            "piv_validity_wm": wm_valid_frac,
            "frames_evaluated": int(min(unity_field.u.shape[0], wm_field.u.shape[0])),
        },
    }
