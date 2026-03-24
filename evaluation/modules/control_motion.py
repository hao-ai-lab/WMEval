from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from ..context import EvaluationContext
from .motion import _center_crop_to_common, _flow_raft


def _get_thresholds(raw_thresholds: Any) -> List[float]:
    if raw_thresholds is None:
        return [1.0, 3.0]
    if isinstance(raw_thresholds, str):
        values = [part.strip() for part in raw_thresholds.split(",")]
        thresholds = [float(value) for value in values if value]
    elif isinstance(raw_thresholds, Iterable):
        thresholds = [float(value) for value in raw_thresholds]
    else:
        thresholds = [float(raw_thresholds)]
    return sorted({value for value in thresholds if value >= 0.0})


def _summarize(values: Sequence[float]) -> Dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _estimate_global_motion(flow: np.ndarray, mode: str) -> np.ndarray:
    flat = flow.reshape(-1, 2)
    if mode == "mean_translation":
        return flat.mean(axis=0).astype(np.float32)
    if mode == "median_translation":
        return np.median(flat, axis=0).astype(np.float32)
    raise ValueError(
        f"Unsupported control global motion mode: {mode!r}. "
        "Use 'affine_ransac', 'median_translation', or 'mean_translation'."
    )


def _angle_error_deg(flow_ref: np.ndarray, flow_gen: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    ref_sel = flow_ref[mask]
    gen_sel = flow_gen[mask]
    denom = np.linalg.norm(ref_sel, axis=1) * np.linalg.norm(gen_sel, axis=1) + 1e-8
    cosine = np.sum(ref_sel * gen_sel, axis=1) / denom
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    return float(np.mean(angle))


def _cosine_similarity(flow_ref: np.ndarray, flow_gen: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    ref_sel = flow_ref[mask]
    gen_sel = flow_gen[mask]
    denom = np.linalg.norm(ref_sel, axis=1) * np.linalg.norm(gen_sel, axis=1) + 1e-8
    cosine = np.sum(ref_sel * gen_sel, axis=1) / denom
    return float(np.mean(np.clip(cosine, -1.0, 1.0)))


def _relative_epe(flow_ref: np.ndarray, flow_gen: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    ref_sel = flow_ref[mask]
    gen_sel = flow_gen[mask]
    numer = np.linalg.norm(gen_sel - ref_sel, axis=1)
    denom = np.linalg.norm(ref_sel, axis=1) + 1.0
    return float(np.mean(numer / denom))


def _fit_affine_motion(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    height, width = flow.shape[:2]
    stride_y = max(1, height // 64)
    stride_x = max(1, width // 64)

    ys = np.arange(0, height, stride_y, dtype=np.float32)
    xs = np.arange(0, width, stride_x, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    src = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1).astype(np.float32)
    sampled_flow = flow[::stride_y, ::stride_x].reshape(-1, 2).astype(np.float32)
    dst = src + sampled_flow

    matrix, _ = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None:
        translation = np.median(flow.reshape(-1, 2), axis=0).astype(np.float32)
        motion_field = np.broadcast_to(
            translation.reshape(1, 1, 2), flow.shape
        ).astype(np.float32)
        return motion_field, None

    full_x, full_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    homogeneous = np.stack(
        [full_x.reshape(-1), full_y.reshape(-1), np.ones(height * width, dtype=np.float32)],
        axis=1,
    )
    dst_full = homogeneous @ matrix.T
    motion_field = dst_full.reshape(height, width, 2) - np.stack(
        [full_x, full_y], axis=2
    )
    return motion_field.astype(np.float32), matrix.astype(np.float32)


def _estimate_camera_motion(
    flow: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, Dict[str, Any]]:
    if mode == "affine_ransac":
        motion_field, matrix = _fit_affine_motion(flow)
        if matrix is None:
            translation = np.median(flow.reshape(-1, 2), axis=0).astype(np.float32)
            return motion_field, {
                "type": "translation_fallback",
                "vector": [float(translation[0]), float(translation[1])],
            }
        return motion_field, {
            "type": "affine_ransac",
            "matrix_2x3": [[float(v) for v in row] for row in matrix.tolist()],
        }

    translation = _estimate_global_motion(flow, mode)
    motion_field = np.broadcast_to(
        translation.reshape(1, 1, 2), flow.shape
    ).astype(np.float32)
    return motion_field, {
        "type": mode,
        "vector": [float(translation[0]), float(translation[1])],
    }


def _compute_metrics(
    flow_ref: np.ndarray,
    flow_gen: np.ndarray,
    motion_mask_threshold: float,
    motion_mask_percentile: float,
    bad_pixel_thresholds: Sequence[float],
    mask_flow: np.ndarray | None = None,
) -> Dict[str, Any]:
    delta = flow_gen - flow_ref
    epe = np.linalg.norm(delta, axis=2)
    mask_source = flow_ref if mask_flow is None else mask_flow
    mag_ref = np.linalg.norm(mask_source, axis=2)
    adaptive_threshold = motion_mask_threshold
    if motion_mask_percentile > 0:
        adaptive_threshold = max(
            adaptive_threshold,
            float(np.percentile(mag_ref, motion_mask_percentile)),
        )
    dynamic_mask = mag_ref > adaptive_threshold

    epe_dynamic = float(np.mean(epe[dynamic_mask])) if np.any(dynamic_mask) else None
    angle_dynamic = _angle_error_deg(flow_ref, flow_gen, dynamic_mask)
    cosine_dynamic = _cosine_similarity(flow_ref, flow_gen, dynamic_mask)
    relative_epe_dynamic = _relative_epe(flow_ref, flow_gen, dynamic_mask)

    bad_pixel_rates: Dict[str, float] = {}
    bad_pixel_rates_dynamic: Dict[str, float | None] = {}
    for threshold in bad_pixel_thresholds:
        key = f"{threshold:g}px"
        bad_pixel_rates[key] = float(np.mean(epe > threshold))
        if np.any(dynamic_mask):
            bad_pixel_rates_dynamic[key] = float(np.mean(epe[dynamic_mask] > threshold))
        else:
            bad_pixel_rates_dynamic[key] = None

    return {
        "epe": float(np.mean(epe)),
        "epe_dynamic": epe_dynamic,
        "angular_error_deg_dynamic": angle_dynamic,
        "cosine_similarity_dynamic": cosine_dynamic,
        "relative_epe_dynamic": relative_epe_dynamic,
        "dynamic_motion_ratio": float(np.mean(dynamic_mask)),
        "dynamic_motion_threshold": adaptive_threshold,
        "bad_pixel_rates": bad_pixel_rates,
        "bad_pixel_rates_dynamic": bad_pixel_rates_dynamic,
    }


def _build_section(
    per_pair_metrics: Sequence[Dict[str, Any]],
    bad_pixel_thresholds: Sequence[float],
) -> Dict[str, Any]:
    epe_values = [item["epe"] for item in per_pair_metrics]
    epe_dynamic_values = [
        float(item["epe_dynamic"])
        for item in per_pair_metrics
        if item["epe_dynamic"] is not None
    ]
    angle_values = [
        float(item["angular_error_deg_dynamic"])
        for item in per_pair_metrics
        if item["angular_error_deg_dynamic"] is not None
    ]
    cosine_values = [
        float(item["cosine_similarity_dynamic"])
        for item in per_pair_metrics
        if item["cosine_similarity_dynamic"] is not None
    ]
    relative_epe_values = [
        float(item["relative_epe_dynamic"])
        for item in per_pair_metrics
        if item["relative_epe_dynamic"] is not None
    ]
    dynamic_ratio_values = [item["dynamic_motion_ratio"] for item in per_pair_metrics]
    dynamic_threshold_values = [item["dynamic_motion_threshold"] for item in per_pair_metrics]

    bad_pixel_summary: Dict[str, Dict[str, Any]] = {}
    for threshold in bad_pixel_thresholds:
        key = f"{threshold:g}px"
        all_values = [item["bad_pixel_rates"][key] for item in per_pair_metrics]
        dynamic_values = [
            float(item["bad_pixel_rates_dynamic"][key])
            for item in per_pair_metrics
            if item["bad_pixel_rates_dynamic"][key] is not None
        ]
        bad_pixel_summary[key] = {
            "per_pair": all_values,
            "summary": _summarize(all_values),
            "dynamic_per_pair": dynamic_values,
            "dynamic_summary": _summarize(dynamic_values),
        }

    return {
        "epe_per_pair": epe_values,
        "epe_summary": _summarize(epe_values),
        "epe_dynamic_per_pair": epe_dynamic_values,
        "epe_dynamic_summary": _summarize(epe_dynamic_values),
        "angular_error_deg_dynamic_per_pair": angle_values,
        "angular_error_deg_dynamic_summary": _summarize(angle_values),
        "cosine_similarity_dynamic_per_pair": cosine_values,
        "cosine_similarity_dynamic_summary": _summarize(cosine_values),
        "relative_epe_dynamic_per_pair": relative_epe_values,
        "relative_epe_dynamic_summary": _summarize(relative_epe_values),
        "dynamic_motion_ratio_per_pair": dynamic_ratio_values,
        "dynamic_motion_ratio_summary": _summarize(dynamic_ratio_values),
        "dynamic_motion_threshold_per_pair": dynamic_threshold_values,
        "dynamic_motion_threshold_summary": _summarize(dynamic_threshold_values),
        "bad_pixel_rates": bad_pixel_summary,
    }


def analyze_control_motion_from_context(
    context: EvaluationContext,
    device: str | None = None,
) -> Dict[str, Any]:
    num_frames = context.num_pairs
    if num_frames < 2:
        return {
            "module": "control_motion",
            "error": "at least 2 aligned frames are required",
        }

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    artifacts = context.artifacts or {}
    motion_mask_threshold = float(artifacts.get("control_motion_mask_threshold", 0.5))
    motion_mask_percentile = float(artifacts.get("control_motion_mask_percentile", 75.0))
    bad_pixel_thresholds = _get_thresholds(
        artifacts.get("control_bad_pixel_thresholds", [1.0, 3.0])
    )
    camera_compensation_enabled = bool(
        artifacts.get("control_camera_compensation", True)
    )
    global_motion_mode = str(
        artifacts.get("control_global_motion_mode", "median_translation")
    )

    weights = Raft_Large_Weights.DEFAULT
    transform = weights.transforms()
    model = raft_large(weights=weights, progress=False).to(device_obj).eval()

    raw_metrics_per_pair: List[Dict[str, Any]] = []
    compensated_metrics_per_pair: List[Dict[str, Any]] = []
    reference_camera_model_per_pair: List[Dict[str, Any]] = []
    candidate_camera_model_per_pair: List[Dict[str, Any]] = []

    for idx in range(1, num_frames):
        ref_prev, gen_prev = _center_crop_to_common(
            context.reference.frames[idx - 1],
            context.candidate.frames[idx - 1],
        )
        ref_next, gen_next = _center_crop_to_common(
            context.reference.frames[idx],
            context.candidate.frames[idx],
        )

        flow_ref = _flow_raft(model, transform, ref_prev, ref_next, device_obj)
        flow_gen = _flow_raft(model, transform, gen_prev, gen_next, device_obj)

        raw_metrics_per_pair.append(
            _compute_metrics(
                flow_ref=flow_ref,
                flow_gen=flow_gen,
                motion_mask_threshold=motion_mask_threshold,
                motion_mask_percentile=motion_mask_percentile,
                bad_pixel_thresholds=bad_pixel_thresholds,
            )
        )

        ref_camera_motion, ref_camera_model = _estimate_camera_motion(
            flow_ref, global_motion_mode
        )
        gen_camera_motion, gen_camera_model = _estimate_camera_motion(
            flow_gen, global_motion_mode
        )
        reference_camera_model_per_pair.append(ref_camera_model)
        candidate_camera_model_per_pair.append(gen_camera_model)

        if camera_compensation_enabled:
            flow_ref_comp = flow_ref - ref_camera_motion
            flow_gen_comp = flow_gen - gen_camera_motion
            compensated_metrics_per_pair.append(
                _compute_metrics(
                    flow_ref=flow_ref_comp,
                    flow_gen=flow_gen_comp,
                    motion_mask_threshold=motion_mask_threshold,
                    motion_mask_percentile=motion_mask_percentile,
                    bad_pixel_thresholds=bad_pixel_thresholds,
                    mask_flow=flow_ref_comp,
                )
            )

    result: Dict[str, Any] = {
        "module": "control_motion",
        "flow_estimator": "torchvision RAFT large",
        "reference_motion_source": "raft_from_reference_video",
        "candidate_motion_source": "raft_from_candidate_video",
        "device": str(device_obj),
        "num_frame_pairs": len(raw_metrics_per_pair),
        "timestamps_sec": context.reference.timestamps_sec[1 : 1 + len(raw_metrics_per_pair)],
        "motion_mask_threshold": motion_mask_threshold,
        "motion_mask_percentile": motion_mask_percentile,
        "bad_pixel_thresholds_px": list(bad_pixel_thresholds),
        "camera_compensation": {
            "enabled": camera_compensation_enabled,
            "mode": global_motion_mode,
            "reference_camera_model_per_pair": reference_camera_model_per_pair,
            "candidate_camera_model_per_pair": candidate_camera_model_per_pair,
        },
        "raw": _build_section(raw_metrics_per_pair, bad_pixel_thresholds),
    }

    if camera_compensation_enabled:
        result["camera_compensated"] = _build_section(
            compensated_metrics_per_pair, bad_pixel_thresholds
        )

    return result
