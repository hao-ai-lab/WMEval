from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from time import perf_counter

from ..context import EvaluationContext
from ..video_io import (
    build_evaluation_context,
    load_frames_by_indices,
    normalized_anchor_indices_from_fractions,
)

try:
    import lpips
except ImportError:
    lpips = None


def _log_lpips(stage: str, **fields: Any) -> None:
    payload = {
        "component": "lpips",
        "stage": stage,
        "ts": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        **fields,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def ensure_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def center_crop_to_common(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])

    def crop(img: np.ndarray) -> np.ndarray:
        y0 = max(0, (img.shape[0] - height) // 2)
        x0 = max(0, (img.shape[1] - width) // 2)
        return img[y0 : y0 + height, x0 : x0 + width]

    return crop(a), crop(b)


def frame_bgr_to_lpips_tensor(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0).to(device)


def summarize_scores(scores: Sequence[float]) -> Dict[str, Optional[float]]:
    if not scores:
        return {
            "lpips_mean": None,
            "lpips_median": None,
            "lpips_std": None,
            "lpips_min": None,
            "lpips_max": None,
        }

    values = np.asarray(scores, dtype=np.float32)
    return {
        "lpips_mean": float(np.mean(values)),
        "lpips_median": float(np.median(values)),
        "lpips_std": float(np.std(values)),
        "lpips_min": float(np.min(values)),
        "lpips_max": float(np.max(values)),
    }


def compute_lpips_scores(
    reference_frames: Sequence[np.ndarray],
    candidate_frames: Sequence[np.ndarray],
    device: torch.device,
) -> List[float]:
    if lpips is None:
        raise RuntimeError("lpips is not installed. Install it with: pip install lpips")

    num_pairs = min(len(reference_frames), len(candidate_frames))
    if num_pairs == 0:
        return []

    _log_lpips(
        "model_load_start",
        device=str(device),
        num_pairs=num_pairs,
    )
    model_load_start = perf_counter()
    loss_fn = lpips.LPIPS(net="vgg").to(device).eval()
    _log_lpips(
        "model_load_done",
        device=str(device),
        elapsed_sec=round(perf_counter() - model_load_start, 3),
    )
    scores: List[float] = []

    score_start = perf_counter()
    with torch.no_grad():
        for idx in range(num_pairs):
            reference_frame = reference_frames[idx]
            candidate_frame = candidate_frames[idx]
            if reference_frame.shape[:2] != candidate_frame.shape[:2]:
                reference_frame, candidate_frame = center_crop_to_common(reference_frame, candidate_frame)

            reference_tensor = frame_bgr_to_lpips_tensor(reference_frame, device)
            candidate_tensor = frame_bgr_to_lpips_tensor(candidate_frame, device)
            distance = loss_fn(reference_tensor, candidate_tensor)
            scores.append(float(distance.mean().item()))
            if idx == 0 or idx == num_pairs - 1 or (idx + 1) % 8 == 0:
                _log_lpips(
                    "score_progress",
                    completed_pairs=idx + 1,
                    num_pairs=num_pairs,
                )

    _log_lpips(
        "score_done",
        elapsed_sec=round(perf_counter() - score_start, 3),
        num_pairs=num_pairs,
    )

    return scores


def _base_lpips_fields(
    context: EvaluationContext,
    device_obj: torch.device,
) -> Dict[str, Any]:
    return {
        "module": "lpips",
        "reference_video": context.reference_video,
        "candidate_video": context.candidate_video,
        "reference_metadata": context.reference.metadata.to_dict(),
        "candidate_metadata": context.candidate.metadata.to_dict(),
        "shared_duration_sec": context.shared_duration_sec,
        "eval_fps": context.eval_fps,
        "max_pairs": context.max_pairs,
        "alignment_mode": context.alignment_mode,
        "device": str(device_obj),
    }


def _analyze_lpips_dense(context: EvaluationContext, device_obj: torch.device) -> Dict[str, Any]:
    if context.num_pairs == 0:
        return {
            "module": "lpips",
            "error": "no overlapping timestamps available",
            "lpips_sampling": "dense",
            **_base_lpips_fields(context, device_obj),
        }

    scores = compute_lpips_scores(
        reference_frames=context.reference.frames[: context.num_pairs],
        candidate_frames=context.candidate.frames[: context.num_pairs],
        device=device_obj,
    )
    return {
        **_base_lpips_fields(context, device_obj),
        "lpips_sampling": "dense",
        "num_pairs": len(scores),
        "timestamps_sec": context.reference.timestamps_sec[: len(scores)],
        "reference_frame_indices": context.reference.frame_indices[: len(scores)],
        "candidate_frame_indices": context.candidate.frame_indices[: len(scores)],
        "lpips_per_pair": scores,
        **summarize_scores(scores),
    }


def _parse_fractions(artifacts: Dict[str, Any]) -> List[float]:
    raw = artifacts.get("lpips_temporal_fractions")
    if raw is None:
        return [0.0, 0.5, 1.0]
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return [float(p) for p in parts]
    return [float(x) for x in raw]


def _analyze_lpips_temporal(context: EvaluationContext, device_obj: torch.device) -> Dict[str, Any]:
    artifacts = context.artifacts or {}
    try:
        fractions = _parse_fractions(artifacts)
    except (TypeError, ValueError) as exc:
        return {
            **_base_lpips_fields(context, device_obj),
            "lpips_sampling": "temporal",
            "error": f"invalid lpips_temporal_fractions: {exc}",
        }
    if not fractions:
        return {
            **_base_lpips_fields(context, device_obj),
            "lpips_sampling": "temporal",
            "error": "lpips_temporal_fractions is empty",
        }

    ref_meta = context.reference.metadata
    cand_meta = context.candidate.metadata
    if ref_meta.frame_count <= 0 or cand_meta.frame_count <= 0:
        return {
            **_base_lpips_fields(context, device_obj),
            "lpips_sampling": "temporal",
            "error": "video has no frames",
        }

    ref_idx = normalized_anchor_indices_from_fractions(ref_meta.frame_count, fractions)
    cand_idx = normalized_anchor_indices_from_fractions(cand_meta.frame_count, fractions)
    ref_frames = load_frames_by_indices(context.reference_video, ref_idx)
    cand_frames = load_frames_by_indices(context.candidate_video, cand_idx)
    n = min(len(ref_frames), len(cand_frames), len(fractions))
    ref_frames = ref_frames[:n]
    cand_frames = cand_frames[:n]
    fractions_used = fractions[:n]
    ref_idx = ref_idx[:n]
    cand_idx = cand_idx[:n]

    labels = artifacts.get("lpips_temporal_labels")
    if isinstance(labels, list) and len(labels) >= n:
        anchor_labels = [str(labels[i]) for i in range(n)]
    else:
        anchor_labels = [f"anchor_{i}" for i in range(n)]

    scores = compute_lpips_scores(ref_frames, cand_frames, device_obj)
    out: Dict[str, Any] = {
        **_base_lpips_fields(context, device_obj),
        "lpips_sampling": "temporal",
        "lpips_temporal_fractions": list(fractions_used),
        "lpips_temporal_labels": anchor_labels,
        "num_pairs": len(scores),
        "reference_frame_indices": ref_idx,
        "candidate_frame_indices": cand_idx,
        "lpips_per_pair": scores,
        **summarize_scores(scores),
    }

    if artifacts.get("save_inspection_artifacts"):
        try:
            from ..inspection_artifacts import write_temporal_inspection

            write_temporal_inspection(
                artifacts=artifacts,
                context=context,
                fractions=fractions_used,
                ref_indices=ref_idx,
                cand_indices=cand_idx,
                scores=scores,
                ref_frames=ref_frames,
                cand_frames=cand_frames,
            )
        except Exception as exc:
            out["inspection_write_warning"] = str(exc)

    return out


def analyze_lpips_from_context(
    context: EvaluationContext,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    device_obj = ensure_device(device)
    artifacts = context.artifacts if context.artifacts is not None else {}
    mode = str(artifacts.get("lpips_sampling", "temporal")).lower().strip()

    if mode in ("dense", "uniform"):
        return _analyze_lpips_dense(context, device_obj)
    if mode == "temporal":
        return _analyze_lpips_temporal(context, device_obj)
    if mode == "causal_vlm_collision":
        from .lpips_causal_vlm import analyze_causal_vlm_lpips

        return analyze_causal_vlm_lpips(context, device_obj, artifacts)
    return {
        "module": "lpips",
        "error": f"unsupported lpips_sampling mode: {mode!r}",
        **_base_lpips_fields(context, device_obj),
    }


def compute_lpips_video_pair(
    reference_video: str,
    candidate_video: str,
    eval_fps: float = 4.0,
    max_pairs: int = 32,
    device: Optional[str] = None,
    text_prompt: str = "",
    lpips_sampling: str = "temporal",
    lpips_temporal_fractions: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    context = build_evaluation_context(
        reference_video=reference_video,
        candidate_video=candidate_video,
        eval_fps=eval_fps,
        max_pairs=max_pairs,
        text_prompt=text_prompt,
    )
    context.artifacts["lpips_sampling"] = lpips_sampling
    if lpips_temporal_fractions is not None:
        context.artifacts["lpips_temporal_fractions"] = list(lpips_temporal_fractions)
    return analyze_lpips_from_context(context=context, device=device)
