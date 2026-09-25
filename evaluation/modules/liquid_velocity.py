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

    # Eq. 14 is a nested expectation, E_t E_{p in V_t}: average over the
    # joint-valid cells within a frame, then over frames. A flat mean over
    # every (t, p) cell would instead weight each frame by how many valid
    # cells it happens to expose. Frames with no valid cells contribute no
    # term.
    usable = np.asarray(joint_valid) & np.isfinite(cos)
    if cos.ndim < 2 or not usable.any():
        flat = cos[usable]
        return float(np.mean(flat)) if flat.size else float("nan")

    per_frame_axes = tuple(range(1, cos.ndim))
    counts = usable.sum(axis=per_frame_axes)
    sums = np.where(usable, cos, 0.0).sum(axis=per_frame_axes)
    frames_with_data = counts > 0
    if not frames_with_data.any():
        return float("nan")
    per_frame = sums[frames_with_data] / counts[frames_with_data]
    return float(np.mean(per_frame))


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
        unity, wm = _lc.resize_piv_to_common_grid(unity, wm)

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


def _subtract_bottle_bulk_velocity(
    field: _lc.PIVField, trace: _lc.BottleTrace
) -> _lc.PIVField:
    """Subtract the bottle's per-frame centroid delta from every PIV vector.

    The PIV field has T-1 frames (one per consecutive pair); the trace has T.
    For pair t we use trace.centroid_xy[t+1] - trace.centroid_xy[t].
    """
    centroids = trace.centroid_xy
    T_field = field.u.shape[0]
    if centroids.shape[0] < T_field + 1:
        raise ValueError(
            f"_subtract_bottle_bulk_velocity: trace has {centroids.shape[0]} frames "
            f"but PIV expects ≥{T_field + 1}"
        )
    deltas = centroids[1 : T_field + 1] - centroids[:T_field]  # (T_field, 2)
    du = deltas[:, 0].astype(np.float32)[:, None, None]
    dv = deltas[:, 1].astype(np.float32)[:, None, None]
    u = field.u - du
    v = field.v - dv
    return _lc.PIVField(u=u, v=v, valid=field.valid)


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
    bottle_prompt = str(arts.get("liquid_bottle_prompt", "the glass bottle"))
    window_size = int(arts.get("liquid_piv_window_size", 32))
    overlap = int(arts.get("liquid_piv_overlap", 16))
    compensate = bool(arts.get("liquid_camera_compensation", True))
    estimator = str(arts.get("liquid_flow_estimator", "raft")).strip().lower()
    if estimator not in ("raft", "openpiv"):
        raise ValueError(
            f"liquid_velocity: liquid_flow_estimator must be 'raft' or 'openpiv', got {estimator!r}"
        )
    min_mask_frac = float(arts.get("liquid_flow_min_mask_frac", 0.5))
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

    if estimator == "raft":
        unity_field = _lc.get_raft_flow_field(
            video_path=context.reference_video, mask=unity_mask,
            window_size=window_size, overlap=overlap, fps=fps,
            cache_dir=cache_root, max_frames=max_pairs,
            min_mask_frac=min_mask_frac,
        )
        wm_field = _lc.get_raft_flow_field(
            video_path=context.candidate_video, mask=wm_mask,
            window_size=window_size, overlap=overlap, fps=fps,
            cache_dir=cache_root, max_frames=max_pairs,
            min_mask_frac=min_mask_frac,
        )
    else:
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

    def _mean_speed(field: _lc.PIVField) -> float:
        u = field.u
        v = field.v
        valid = field.valid & np.isfinite(u) & np.isfinite(v)
        if not valid.any():
            return float("nan")
        return float(np.mean(np.sqrt(u[valid] ** 2 + v[valid] ** 2)))

    def _bulk_speed(trace: _lc.BottleTrace) -> float:
        # Frame-to-frame centroid delta magnitude — what bulk-comp will subtract.
        c = trace.centroid_xy
        if c.shape[0] < 2:
            return float("nan")
        dx = c[1:, 0] - c[:-1, 0]
        dy = c[1:, 1] - c[:-1, 1]
        return float(np.mean(np.sqrt(dx * dx + dy * dy)))

    speed_unity_pre = _mean_speed(unity_field)
    speed_wm_pre = _mean_speed(wm_field)

    bottle_trace_unity_valid = None
    bottle_trace_wm_valid = None
    bottle_bulk_unity = None
    bottle_bulk_wm = None
    if compensate:
        unity_trace = _lc.get_bottle_centroid_trace(
            video_path=context.reference_video, sam2_model=sam2_model,
            bottle_prompt=bottle_prompt, fps=fps, max_pairs=max_pairs,
            cache_dir=cache_root,
        )
        wm_trace = _lc.get_bottle_centroid_trace(
            video_path=context.candidate_video, sam2_model=sam2_model,
            bottle_prompt=bottle_prompt, fps=fps, max_pairs=max_pairs,
            cache_dir=cache_root,
        )
        bottle_bulk_unity = _bulk_speed(unity_trace)
        bottle_bulk_wm = _bulk_speed(wm_trace)
        unity_field = _subtract_bottle_bulk_velocity(unity_field, unity_trace)
        wm_field = _subtract_bottle_bulk_velocity(wm_field, wm_trace)
        bottle_trace_unity_valid = float(np.mean(unity_trace.valid))
        bottle_trace_wm_valid = float(np.mean(wm_trace.valid))

    metrics = compute_velocity_metrics_from_fields(unity=unity_field, wm=wm_field)

    unity_valid_frac = float(np.mean(unity_field.valid))
    wm_valid_frac = float(np.mean(wm_field.valid))

    speed_unity_post = _mean_speed(unity_field)
    speed_wm_post = _mean_speed(wm_field)

    def _mask_diag(masks: np.ndarray) -> dict:
        # Mask too small for the PIV window will produce ~0 valid PIV cells
        # even with perfect texture; this lets us tell mask-size failures
        # apart from texture-correlation failures when piv_validity is low.
        T, H, W = masks.shape
        areas = masks.reshape(T, -1).sum(axis=1).astype(np.float64)
        nonempty = areas > 0
        return {
            "frames": int(T),
            "frames_with_mask": int(nonempty.sum()),
            "mean_area_px": float(areas.mean()),
            "median_area_px": float(np.median(areas)),
            "mean_area_frac": float(areas.mean() / (H * W)) if H * W else 0.0,
        }

    return {
        "module": "liquid_velocity",
        **metrics,
        "diagnostics": {
            "piv_validity_unity": unity_valid_frac,
            "piv_validity_wm": wm_valid_frac,
            "frames_evaluated": int(min(unity_field.u.shape[0], wm_field.u.shape[0])),
            "camera_compensation": compensate,
            "bottle_trace_validity_unity": bottle_trace_unity_valid,
            "bottle_trace_validity_wm": bottle_trace_wm_valid,
            "liquid_mask_unity": _mask_diag(unity_mask.masks),
            "liquid_mask_wm": _mask_diag(wm_mask.masks),
            "piv_window_size": int(window_size),
            "piv_overlap": int(overlap),
            "flow_estimator": estimator,
            "min_mask_frac": float(min_mask_frac),
            "mean_speed_unity_px_per_frame": speed_unity_post,
            "mean_speed_wm_px_per_frame": speed_wm_post,
            "mean_speed_unity_pre_comp_px_per_frame": speed_unity_pre,
            "mean_speed_wm_pre_comp_px_per_frame": speed_wm_pre,
            "bottle_bulk_speed_unity_px_per_frame": bottle_bulk_unity,
            "bottle_bulk_speed_wm_px_per_frame": bottle_bulk_wm,
        },
    }
