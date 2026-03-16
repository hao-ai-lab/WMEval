from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from ..context import EvaluationContext

try:
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception:
    build_sam2_hf = None
    SAM2ImagePredictor = None


def _proxy_shadow_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)
    h, w = v.shape

    v_th = np.percentile(v, 35.0)
    s_th = np.percentile(s, 75.0)
    bottom = np.zeros((h, w), dtype=bool)
    bottom[int(h * 0.35) :, :] = True
    mask = (v <= v_th) & (s <= s_th) & bottom

    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return mask_u8 > 0


def _bbox_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1 = float(np.min(xs))
    y1 = float(np.min(ys))
    x2 = float(np.max(xs))
    y2 = float(np.max(ys))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _sam2_track_masks(frames: List[np.ndarray], predictor: Any, init_bbox: np.ndarray) -> List[np.ndarray]:
    bbox = init_bbox.copy()
    masks: List[np.ndarray] = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(frame_rgb)
        with torch.inference_mode():
            pred_masks, _, _ = predictor.predict(box=bbox[None, :], multimask_output=False)
        if isinstance(pred_masks, torch.Tensor):
            pred_masks = pred_masks.cpu().numpy()
        if pred_masks.ndim == 4:
            pred_masks = pred_masks[:, 0]
        mask = pred_masks[0].astype(bool)
        masks.append(mask)
        new_bbox = _bbox_from_mask(mask)
        if new_bbox is not None:
            bbox = new_bbox
    return masks


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union <= 0:
        return 1.0
    return float(inter / union)


def _centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return np.asarray([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)


def _summarize(values: List[float], prefix: str) -> Dict[str, float | None]:
    if not values:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_p90": None,
            f"{prefix}_max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_max": float(np.max(arr)),
    }


def analyze_shadow_from_context(context: EvaluationContext, device: str | None = None) -> Dict[str, Any]:
    num_frames = context.num_pairs
    if num_frames == 0:
        return {"module": "shadow", "error": "no overlapping timestamps available"}
    if build_sam2_hf is None or SAM2ImagePredictor is None:
        return {
            "module": "shadow",
            "error": "SAM2 is not installed. Install sam2 to enable this module.",
        }

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    sam_model_id = str(context.artifacts.get("sam2_model") or "facebook/sam2.1-hiera-small")
    sam_model = build_sam2_hf(sam_model_id, device=device_obj)
    predictor = SAM2ImagePredictor(sam_model=sam_model)

    ref_frames = context.reference.frames[:num_frames]
    cand_frames = context.candidate.frames[:num_frames]

    init_mask_ref = _proxy_shadow_mask(ref_frames[0])
    init_bbox_ref = _bbox_from_mask(init_mask_ref)
    if init_bbox_ref is None:
        return {
            "module": "shadow",
            "error": "Could not find initial shadow bbox from first frame",
        }

    ref_masks = _sam2_track_masks(ref_frames, predictor, init_bbox_ref)
    cand_masks = _sam2_track_masks(cand_frames, predictor, init_bbox_ref)

    iou_per_frame: List[float] = []
    centroid_error_px: List[float] = []
    for m_ref, m_cand in zip(ref_masks, cand_masks):
        iou_per_frame.append(_iou(m_ref, m_cand))
        c_ref = _centroid(m_ref)
        c_cand = _centroid(m_cand)
        if c_ref is not None and c_cand is not None:
            centroid_error_px.append(float(np.linalg.norm(c_cand - c_ref)))

    return {
        "module": "shadow",
        "shadow_estimator": "SAM2ImagePredictor tracked from first-frame bbox",
        "sam2_model": sam_model_id,
        "device": str(device_obj),
        "num_frames": num_frames,
        "timestamps_sec": context.reference.timestamps_sec[:num_frames],
        "shadow_iou_per_frame": iou_per_frame,
        "shadow_centroid_error_px_per_frame": centroid_error_px,
        **_summarize(iou_per_frame, "shadow_iou"),
        **_summarize(centroid_error_px, "shadow_centroid_error_px"),
    }
