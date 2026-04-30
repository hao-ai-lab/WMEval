"""Frame sampling from generated videos for scene alignment evaluation.

Supports two modes:
  - frame-count mode (default): take first N frames from the video, uniformly
    subsample to K frames. Best for initial-scene evaluation because it's
    FPS-invariant and avoids dynamics contamination.
  - time-based mode: sample from first N seconds at a given FPS.

Heavy dependencies (cv2, numpy, evaluation.video_io) are imported lazily
so that other modules can import utility functions without triggering them.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np


def _ensure_project_root() -> None:
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)


def _subsample_indices(frame_window: int, num_samples: int) -> List[int]:
    """Uniformly pick num_samples indices from [0, frame_window).

    Examples:
        _subsample_indices(12, 4) -> [0, 4, 8, 11]
        _subsample_indices(12, 3) -> [0, 5, 11]
        _subsample_indices(5, 5)  -> [0, 1, 2, 3, 4]
    """
    if frame_window <= 0 or num_samples <= 0:
        return []
    if num_samples >= frame_window:
        return list(range(frame_window))
    if num_samples == 1:
        return [0]
    step = (frame_window - 1) / (num_samples - 1)
    return [round(i * step) for i in range(num_samples)]


def _time_based_indices(
    fps: float,
    total_frames: int,
    duration_sec: float,
    sample_fps: float = 1.0,
    max_frames: int = 10,
    start_seconds: float = 0.0,
) -> List[int]:
    """Compute frame indices for uniform sampling from [start, start+duration).

    Parameters
    ----------
    start_seconds : float
        Offset in seconds from the start of the video (default 0.0).
        Backward-compatible: existing callers that omit this get v1 behavior.
    """
    if fps <= 0 or total_frames <= 0 or duration_sec <= 0:
        return []

    step = 1.0 / sample_fps
    timestamps: List[float] = []
    t = start_seconds
    end = start_seconds + duration_sec
    while t < end and len(timestamps) < max_frames:
        timestamps.append(t)
        t += step

    indices: List[int] = []
    for ts in timestamps:
        idx = int(round(ts * fps))
        idx = min(max(idx, 0), total_frames - 1)
        indices.append(idx)
    return indices


def encode_frame_b64(
    frame: "np.ndarray",
    fmt: str = ".jpg",
    quality: int = 85,
) -> str:
    """Encode a BGR numpy frame to base64 JPEG/PNG string."""
    import cv2

    params: list = []
    if fmt == ".jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buf = cv2.imencode(fmt, frame, params)
    if not ok:
        raise RuntimeError("Failed to encode frame")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_image_file_b64(path: str) -> str:
    """Read an image file and return base64 string."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def sample_video_frames(
    video_path: str,
    *,
    first_n_frames: int = 12,
    num_samples: int = 4,
    first_n_seconds: Optional[float] = None,
    sample_fps: float = 1.0,
    max_frames: int = 10,
    start_seconds: float = 0.0,
    start_frame: int = 0,
) -> Tuple[List["np.ndarray"], List[float], List[int]]:
    """Sample frames from a video for scene evaluation.

    Two modes:
      1. Frame-count mode (default): take `first_n_frames` frames starting at
         `start_frame`, then uniformly subsample to `num_samples`.
      2. Time-based mode: activated when `first_n_seconds` is not None.
         Samples at `sample_fps` within [start_seconds, start_seconds + N).

    Parameters
    ----------
    start_seconds : float
        Time-based mode offset (default 0.0, backward-compatible).
    start_frame : int
        Frame-count mode offset (default 0, backward-compatible).

    Returns
    -------
    frames : list of np.ndarray (BGR)
    timestamps_sec : list of float
    frame_indices : list of int
    """
    _ensure_project_root()
    from evaluation.video_io import get_video_metadata, load_frames_by_indices

    meta = get_video_metadata(video_path)
    fps = meta.fps if meta.fps > 0 else 1.0

    if first_n_seconds is not None:
        video_duration = meta.duration_sec or 0.0
        remaining = max(0.0, video_duration - start_seconds)
        effective_duration = min(first_n_seconds, remaining)
        indices = _time_based_indices(
            fps=fps,
            total_frames=meta.frame_count,
            duration_sec=effective_duration,
            sample_fps=sample_fps,
            max_frames=max_frames,
            start_seconds=start_seconds,
        )
    else:
        available = max(0, meta.frame_count - start_frame)
        window = min(first_n_frames, available)
        sub = _subsample_indices(window, num_samples)
        indices = [i + start_frame for i in sub]

    if not indices:
        return [], [], []

    frames = load_frames_by_indices(video_path, indices)

    timestamps = [idx / fps for idx in indices[: len(frames)]]
    used_indices = indices[: len(frames)]

    return frames, timestamps, used_indices


def prepare_frames_for_vlm(
    video_path: str,
    *,
    first_n_frames: int = 12,
    num_samples: int = 4,
    first_n_seconds: Optional[float] = None,
    sample_fps: float = 1.0,
    max_frames: int = 10,
    start_seconds: float = 0.0,
    start_frame: int = 0,
    gt_frame_path: Optional[str] = None,
    label_prefix: str = "sampled_frame",
    use_timestamp_labels: bool = False,
) -> Tuple[List[Tuple[str, str]], int]:
    """Sample video frames and encode them for VLM API calls.

    Parameters
    ----------
    start_seconds / start_frame : offset parameters (default 0, backward-compatible).
    label_prefix : prefix for frame labels.
    use_timestamp_labels : if True, label each frame as "{prefix}_t={ts:.1f}s"
        instead of "{prefix}_{i}". Requires time-based mode (first_n_seconds).

    Returns
    -------
    images_b64 : list of (label, base64_string) tuples
        GT frame (if any) comes first, then sampled frames.
    num_candidate_frames : int
        Number of sampled video frames (excludes GT frame).
    """
    images_b64: List[Tuple[str, str]] = []

    if gt_frame_path and Path(gt_frame_path).is_file():
        images_b64.append(("gt_initial_frame", encode_image_file_b64(gt_frame_path)))

    frames, timestamps, indices = sample_video_frames(
        video_path,
        first_n_frames=first_n_frames,
        num_samples=num_samples,
        first_n_seconds=first_n_seconds,
        sample_fps=sample_fps,
        max_frames=max_frames,
        start_seconds=start_seconds,
        start_frame=start_frame,
    )

    for i, frame in enumerate(frames):
        if use_timestamp_labels and i < len(timestamps):
            label = f"{label_prefix}_t={timestamps[i]:.1f}s"
        else:
            label = f"{label_prefix}_{i}"
        images_b64.append((label, encode_frame_b64(frame)))

    return images_b64, len(frames)
