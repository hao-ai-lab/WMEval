from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from ..context import EvaluationContext
from .flow_divergence_metrics import (
    compute_frame_metrics,
    compute_temporal_metrics,
    json_safe_floats,
)
from .flow_utils import (
    center_crop_to_common as _center_crop_to_common,
    estimate_camera_motion as _estimate_camera_motion,
    flow_raft as _flow_raft,
    load_raft_model,
)


VENDORED_SYNTHETIC_DIR = Path(__file__).resolve().parent / "vendor" / "ptlflow_synthetic"


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


def _import_synthetic_flow_generator(ptlflow_root: str) -> Tuple[Any, Any]:
    root = Path(ptlflow_root).resolve()
    mod_path = root / "synthetic_flow.py"
    if not mod_path.is_file():
        raise FileNotFoundError(f"synthetic_flow.py not found under {root}")
    spec = importlib.util.spec_from_file_location("wm_synthetic_flow", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load synthetic_flow from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.SyntheticFlowGenerator, mod.load_calibration


def _scale_calibration(calibration: Dict[str, Any], h: int, w: int) -> Dict[str, Any]:
    del h  # height unused; focal length scaled by width like pinhole fx
    out = dict(calibration)
    fs = out.get("frame_shape", [352, 640])
    if isinstance(fs, (list, tuple)) and len(fs) == 2:
        _, fw = int(fs[0]), int(fs[1])
    else:
        fw = 640
    f = float(out["focal_length"])
    out["focal_length"] = f * (float(w) / float(fw))
    return out


def _load_action_npy(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True).item()
    if "keyboard" not in data or "mouse" not in data:
        raise ValueError(f"action npy must contain 'keyboard' and 'mouse' keys: {path}")
    return data


def _infer_actions_coordinate_descent(
    gen: Any,
    flow_target: np.ndarray,
    frame_bgr: np.ndarray,
    num_rounds: int = 48,
    initial_step: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit continuous keyboard(6)+mouse(2) to approximate RAFT flow with synthetic field."""

    h, w = flow_target.shape[:2]
    sy = max(1, h // 24)
    sx = max(1, w // 24)
    tgt = flow_target[::sy, ::sx, :].astype(np.float64)

    def mse_kb_ms(kb: np.ndarray, ms: np.ndarray) -> float:
        syn = gen.generate_flow(
            kb.astype(np.float32),
            ms.astype(np.float32),
            frame_bgr=frame_bgr,
        )
        s = syn[::sy, ::sx, :].astype(np.float64)
        return float(np.mean((s - tgt) ** 2))

    x = np.zeros(8, dtype=np.float32)
    best = mse_kb_ms(x[:6], x[6:8])
    step = initial_step
    for _ in range(num_rounds):
        improved = False
        for j in range(8):
            for sign in (-1.0, 1.0):
                delta = np.zeros(8, dtype=np.float32)
                delta[j] = sign * step
                kb2 = np.clip(x[:6] + delta[:6], -2.0, 2.0)
                ms2 = np.clip(x[6:8] + delta[6:8], -2.0, 2.0)
                err = mse_kb_ms(kb2, ms2)
                if err < best - 1e-10:
                    best = err
                    x[:6] = kb2
                    x[6:8] = ms2
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-3:
                break
    return x[:6].copy(), x[6:8].copy()


def _metric_dict_for_temporal(row: Dict[str, Any]) -> Dict[str, float]:
    skip = {"pair_index", "candidate_frame_index", "inferred_actions"}
    out: Dict[str, float] = {}
    for k, v in row.items():
        if k in skip:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and np.isfinite(v):
            out[k] = float(v)
    return out


def _build_synthetic_branch(
    context: EvaluationContext,
    flow_refs: List[np.ndarray],
    flow_gens: List[np.ndarray],
    device_obj: torch.device,
    *,
    ptlflow_root: str,
    calibration_path: str,
    action_path: Optional[str],
    infer_actions: bool,
    use_depth: bool,
    synthetic_grid_size: int,
) -> Dict[str, Any]:
    del device_obj
    if len(flow_refs) != len(flow_gens):
        return {
            "enabled": False,
            "error": "internal: flow list length mismatch",
        }

    SyntheticFlowGenerator, load_calibration = _import_synthetic_flow_generator(ptlflow_root)
    calibration = load_calibration(calibration_path)
    num_frames = context.num_pairs
    if num_frames < 2:
        return {"enabled": False, "error": "not enough frames"}

    _, gen_prev0 = _center_crop_to_common(
        context.reference.frames[0],
        context.candidate.frames[0],
    )
    h, w = gen_prev0.shape[:2]
    cal_scaled = _scale_calibration(calibration, h, w)

    gen_model = SyntheticFlowGenerator(
        cal_scaled,
        frame_shape=(h, w),
        use_depth=use_depth,
    )
    if use_depth:
        try:
            gen_model.load_depth_model()
        except Exception as exc:
            return {
                "enabled": False,
                "error": f"synthetic_use_depth requires transformers & depth model: {exc}",
            }

    actions_loaded: Optional[Dict[str, np.ndarray]] = None
    idx_needed = context.candidate.frame_indices[: num_frames - 1]
    max_idx_needed = max(idx_needed) if idx_needed else 0

    if action_path:
        actions_loaded = _load_action_npy(action_path)
        kb_len = len(actions_loaded["keyboard"])
        if kb_len <= max_idx_needed:
            return {
                "enabled": False,
                "error": (
                    f"action length {kb_len} <= max sampled frame index {max_idx_needed}; "
                    "use a full-sequence action npy or enable synthetic_infer_actions"
                ),
            }

    per_pair: List[Dict[str, Any]] = []
    n_pairs = len(flow_gens)

    for p in range(n_pairs):
        gen_prev, _ = _center_crop_to_common(
            context.reference.frames[p],
            context.candidate.frames[p],
        )
        frame_idx = int(context.candidate.frame_indices[p])

        if actions_loaded is not None:
            kb = actions_loaded["keyboard"][frame_idx].astype(np.float32)
            ms = actions_loaded["mouse"][frame_idx].astype(np.float32)
            flow_synth = gen_model.generate_flow(kb, ms, frame_bgr=gen_prev)
            inferred = False
        elif infer_actions:
            kb, ms = _infer_actions_coordinate_descent(
                gen_model,
                flow_refs[p],
                gen_prev,
            )
            flow_synth = gen_model.generate_flow(kb, ms, frame_bgr=gen_prev)
            inferred = True
        else:
            return {
                "enabled": False,
                "error": "provide synthetic_action_file or enable synthetic_infer_actions",
            }

        metrics = compute_frame_metrics(
            flow_synth,
            flow_gens[p],
            grid_size=synthetic_grid_size,
        )
        metrics["pair_index"] = p
        metrics["candidate_frame_index"] = frame_idx
        metrics["inferred_actions"] = inferred
        per_pair.append(json_safe_floats(metrics))

    temporal = compute_temporal_metrics(
        [_metric_dict_for_temporal(m) for m in per_pair]
    )
    temporal_clean: Dict[str, Any] = {}
    for k, v in temporal.items():
        if isinstance(v, float) and not np.isfinite(v):
            temporal_clean[k] = None
        else:
            temporal_clean[k] = v

    highlight_keys = (
        "mf_epe_mean",
        "mf_angle_err_mean",
        "mf_cosine_mean",
        "pixel_epe_mean_mean",
        "fl_all_mean",
        "flow_kl_2d_mean",
        "divergence_onset_frame",
    )

    return {
        "enabled": True,
        "reference_motion_source": (
            "longuet_higgins_synthetic_from_actions"
            if actions_loaded is not None
            else "longuet_higgins_synthetic_inferred_from_raft_reference"
        ),
        "candidate_motion_source": "raft_from_candidate_video",
        "flow_estimator": "torchvision RAFT large",
        "ptlflow_root": str(Path(ptlflow_root).resolve()),
        "calibration": str(Path(calibration_path).resolve()),
        "action_file": str(Path(action_path).resolve()) if action_path else None,
        "synthetic_use_depth": use_depth,
        "synthetic_infer_actions": infer_actions and actions_loaded is None,
        "per_pair_metrics": per_pair,
        "temporal_summary": temporal_clean,
        "highlights": {k: temporal_clean[k] for k in highlight_keys if k in temporal_clean},
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
    synthetic_calibration = artifacts.get("synthetic_calibration") or str(
        VENDORED_SYNTHETIC_DIR / "calibration.json"
    )
    synthetic_action_file = artifacts.get("synthetic_action_file")
    synthetic_infer_actions = bool(artifacts.get("synthetic_infer_actions", False))
    synthetic_use_depth = bool(artifacts.get("synthetic_use_depth", False))
    ptlflow_root = str(artifacts.get("ptlflow_root") or VENDORED_SYNTHETIC_DIR)
    synthetic_grid_size = int(artifacts.get("synthetic_grid_size", 8))

    model, transform = load_raft_model(device_obj)

    raw_metrics_per_pair: List[Dict[str, Any]] = []
    compensated_metrics_per_pair: List[Dict[str, Any]] = []
    reference_camera_model_per_pair: List[Dict[str, Any]] = []
    candidate_camera_model_per_pair: List[Dict[str, Any]] = []
    flow_refs: List[np.ndarray] = []
    flow_gens: List[np.ndarray] = []

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
        flow_refs.append(flow_ref)
        flow_gens.append(flow_gen)

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

    run_synthetic = bool(synthetic_calibration) and (
        bool(synthetic_action_file) or synthetic_infer_actions
    )
    if run_synthetic:
        try:
            result["synthetic_vs_candidate"] = _build_synthetic_branch(
                context,
                flow_refs,
                flow_gens,
                device_obj,
                ptlflow_root=ptlflow_root,
                calibration_path=str(synthetic_calibration),
                action_path=str(synthetic_action_file) if synthetic_action_file else None,
                infer_actions=synthetic_infer_actions,
                use_depth=synthetic_use_depth,
                synthetic_grid_size=synthetic_grid_size,
            )
        except Exception as exc:
            result["synthetic_vs_candidate"] = {
                "enabled": False,
                "error": str(exc),
            }
    else:
        result["synthetic_vs_candidate"] = {
            "enabled": False,
            "note": (
                "set artifacts synthetic_calibration and (synthetic_action_file or "
                "synthetic_infer_actions) to run Longuet-Higgins synthetic flow vs RAFT candidate"
            ),
        }

    return result
