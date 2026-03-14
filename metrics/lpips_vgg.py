from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch

try:
    import lpips
except ImportError:
    lpips = None

from .video_io import build_sample_timestamps, get_video_metadata, sample_video_frames


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


def compute_lpips_video_pair(
    reference_video: str,
    candidate_video: str,
    eval_fps: float = 4.0,
    max_pairs: int = 32,
    device: str | None = None,
) -> Dict[str, Any]:
    reference_metadata = get_video_metadata(reference_video)
    candidate_metadata = get_video_metadata(candidate_video)
    device_obj = ensure_device(device)

    if reference_metadata.frame_count <= 0:
        raise RuntimeError(f"Reference video has no frames: {reference_video}")
    if candidate_metadata.frame_count <= 0:
        raise RuntimeError(f"Candidate video has no frames: {candidate_video}")

    reference_duration = reference_metadata.duration_sec or 0.0
    candidate_duration = candidate_metadata.duration_sec or 0.0
    shared_duration_sec = min(reference_duration, candidate_duration)
    timestamps_sec = build_sample_timestamps(
        duration_sec=shared_duration_sec,
        eval_fps=eval_fps,
        max_pairs=max_pairs,
    )

    if not timestamps_sec:
        return {
            "metric": "lpips_vgg",
            "error": "no overlapping timestamps available",
            "reference_video": reference_video,
            "candidate_video": candidate_video,
            "reference_metadata": reference_metadata.to_dict(),
            "candidate_metadata": candidate_metadata.to_dict(),
            "eval_fps": eval_fps,
            "max_pairs": max_pairs,
            "device": str(device_obj),
        }

    reference_sample = sample_video_frames(reference_video, timestamps_sec)
    candidate_sample = sample_video_frames(candidate_video, timestamps_sec)
    used_count = min(
        len(reference_sample["frames"]),
        len(candidate_sample["frames"]),
        len(timestamps_sec),
    )
    scores = compute_lpips_scores(
        reference_frames=reference_sample["frames"][:used_count],
        candidate_frames=candidate_sample["frames"][:used_count],
        device=device_obj,
    )

    return {
        "metric": "lpips_vgg",
        "reference_video": reference_video,
        "candidate_video": candidate_video,
        "reference_metadata": reference_metadata.to_dict(),
        "candidate_metadata": candidate_metadata.to_dict(),
        "shared_duration_sec": shared_duration_sec,
        "eval_fps": eval_fps,
        "max_pairs": max_pairs,
        "device": str(device_obj),
        "num_pairs": len(scores),
        "timestamps_sec": reference_sample["timestamps_sec"][:len(scores)],
        "reference_frame_indices": reference_sample["frame_indices"][:len(scores)],
        "candidate_frame_indices": candidate_sample["frame_indices"][:len(scores)],
        "lpips_per_pair": scores,
        **summarize_scores(scores),
    }
