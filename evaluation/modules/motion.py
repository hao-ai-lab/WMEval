from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from ..context import EvaluationContext


def _center_crop_to_common(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])

    def crop(img: np.ndarray) -> np.ndarray:
        y0 = (img.shape[0] - h) // 2
        x0 = (img.shape[1] - w) // 2
        return img[y0 : y0 + h, x0 : x0 + w]

    return crop(a), crop(b)


def _to_rgb_tchw(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    frame_rgb = frame_bgr[..., ::-1].copy()
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0)
    return tensor.to(device)


def _flow_raft(model: torch.nn.Module, transform, prev_bgr: np.ndarray, next_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    t0 = _to_rgb_tchw(prev_bgr, device)
    t1 = _to_rgb_tchw(next_bgr, device)
    t0, t1 = transform(t0, t1)

    h, w = t0.shape[-2:]
    target_h = max(128, int(np.ceil(h / 8.0) * 8))
    target_w = max(128, int(np.ceil(w / 8.0) * 8))
    if target_h != h or target_w != w:
        t0 = F.interpolate(t0, size=(target_h, target_w), mode="bilinear", align_corners=False)
        t1 = F.interpolate(t1, size=(target_h, target_w), mode="bilinear", align_corners=False)

    with torch.no_grad():
        flow_list = model(t0, t1)
    flow_t = flow_list[-1][0:1]

    if flow_t.shape[-2:] != (h, w):
        flow_t = F.interpolate(flow_t, size=(h, w), mode="bilinear", align_corners=False)
        flow_t[:, 0] *= float(w) / float(target_w)
        flow_t[:, 1] *= float(h) / float(target_h)

    flow = flow_t[0].detach().cpu().permute(1, 2, 0).numpy()
    return flow.astype(np.float32)


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


def analyze_motion_from_context(context: EvaluationContext, device: str | None = None) -> Dict[str, Any]:
    num_frames = context.num_pairs
    if num_frames < 2:
        return {"module": "motion", "error": "at least 2 aligned frames are required"}

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    weights = Raft_Large_Weights.DEFAULT
    transform = weights.transforms()
    model = raft_large(weights=weights, progress=False).to(device_obj).eval()

    epe_all: List[float] = []
    epe_dynamic: List[float] = []
    flow_angle_error_deg: List[float] = []
    dynamic_ratio: List[float] = []

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

        delta = flow_gen - flow_ref
        epe = np.linalg.norm(delta, axis=2)
        epe_all.append(float(np.mean(epe)))

        mag_ref = np.linalg.norm(flow_ref, axis=2)
        motion_mask = mag_ref > 0.5
        dynamic_ratio.append(float(np.mean(motion_mask)))
        if np.any(motion_mask):
            epe_dynamic.append(float(np.mean(epe[motion_mask])))

        denom = np.linalg.norm(flow_ref, axis=2) * np.linalg.norm(flow_gen, axis=2) + 1e-8
        cosine = np.sum(flow_ref * flow_gen, axis=2) / denom
        cosine = np.clip(cosine, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))
        flow_angle_error_deg.append(float(np.mean(angle)))

    return {
        "module": "motion",
        "flow_estimator": "torchvision RAFT large",
        "device": str(device_obj),
        "num_frame_pairs": len(epe_all),
        "timestamps_sec": context.reference.timestamps_sec[1 : 1 + len(epe_all)],
        "epe_all_per_pair": epe_all,
        "epe_dynamic_per_pair": epe_dynamic,
        "flow_angle_error_deg_per_pair": flow_angle_error_deg,
        "dynamic_motion_ratio_per_pair": dynamic_ratio,
        **_summarize(epe_all, "epe_all"),
        **_summarize(epe_dynamic, "epe_dynamic"),
        **_summarize(flow_angle_error_deg, "flow_angle_error_deg"),
        **_summarize(dynamic_ratio, "dynamic_motion_ratio"),
    }
