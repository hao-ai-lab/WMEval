from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from ..context import EvaluationContext
from .flow_utils import center_crop_to_common as _center_crop_to_common
from .flow_utils import flow_raft as _flow_raft
from .flow_utils import load_raft_model


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

    model, transform = load_raft_model(device_obj)

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
