from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch

try:
    import lpips
except ImportError:
    lpips = None

from .video_io import EvaluationContext, build_evaluation_context


def ensure_device(device: str | None = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def center_crop_to_common(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])

    def crop(img: np.ndarray) -> np.ndarray:
        y0 = max(0, (img.shape[0] - height) // 2)
        x0 = max(0, (img.shape[1] - width) // 2)
        return img[y0:y0 + height, x0:x0 + width]

    return crop(a), crop(b)


def frame_bgr_to_lpips_tensor(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0).to(device)


def summarize_scores(scores: Sequence[float]) -> Dict[str, float | None]:
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

    loss_fn = lpips.LPIPS(net="vgg").to(device).eval()
    scores: List[float] = []

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

    return scores


def analyze_lpips_from_context(
    context: EvaluationContext,
    device: str | None = None,
) -> Dict[str, Any]:
    device_obj = ensure_device(device)

    if context.num_pairs == 0:
        return {
            "module": "lpips",
            "error": "no overlapping timestamps available",
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

    scores = compute_lpips_scores(
        reference_frames=context.reference.frames[:context.num_pairs],
        candidate_frames=context.candidate.frames[:context.num_pairs],
        device=device_obj,
    )
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
        "num_pairs": len(scores),
        "timestamps_sec": context.reference.timestamps_sec[:len(scores)],
        "reference_frame_indices": context.reference.frame_indices[:len(scores)],
        "candidate_frame_indices": context.candidate.frame_indices[:len(scores)],
        "lpips_per_pair": scores,
        **summarize_scores(scores),
    }


def compute_lpips_video_pair(
    reference_video: str,
    candidate_video: str,
    eval_fps: float = 4.0,
    max_pairs: int = 32,
    device: str | None = None,
    text_prompt: str = "",
) -> Dict[str, Any]:
    context = build_evaluation_context(
        reference_video=reference_video,
        candidate_video=candidate_video,
        eval_fps=eval_fps,
        max_pairs=max_pairs,
        text_prompt=text_prompt,
    )
    return analyze_lpips_from_context(context=context, device=device)
