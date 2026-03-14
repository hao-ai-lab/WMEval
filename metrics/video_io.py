from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float | None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_video_metadata(video_path: str) -> VideoMetadata:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration_sec = None
    if fps > 0 and frame_count > 0:
        duration_sec = frame_count / fps

    return VideoMetadata(
        path=str(Path(video_path)),
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_sec=duration_sec,
    )


def build_sample_timestamps(
    duration_sec: float,
    eval_fps: float,
    max_pairs: int | None = None,
) -> List[float]:
    if duration_sec <= 0:
        return []
    if eval_fps <= 0:
        raise ValueError(f"eval_fps must be > 0, got {eval_fps}")

    step = 1.0 / eval_fps
    timestamps: List[float] = []
    current_t = 0.0
    while current_t < duration_sec:
        timestamps.append(current_t)
        if max_pairs is not None and len(timestamps) >= max_pairs:
            break
        current_t += step
    return timestamps


def timestamp_to_frame_index(timestamp_sec: float, metadata: VideoMetadata) -> int:
    if metadata.frame_count <= 0:
        raise RuntimeError(f"Video has no frames: {metadata.path}")
    if metadata.fps <= 0:
        return min(metadata.frame_count - 1, int(round(timestamp_sec)))

    frame_idx = int(round(timestamp_sec * metadata.fps))
    return min(max(frame_idx, 0), metadata.frame_count - 1)


def load_frames_by_indices(video_path: str, frame_indices: Sequence[int]) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: List[np.ndarray] = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()
    return frames


def sample_video_frames(
    video_path: str,
    timestamps_sec: Sequence[float],
) -> Dict[str, Any]:
    metadata = get_video_metadata(video_path)
    frame_indices = [timestamp_to_frame_index(t, metadata) for t in timestamps_sec]
    frames = load_frames_by_indices(video_path, frame_indices)

    used_count = min(len(frames), len(frame_indices), len(timestamps_sec))
    return {
        "metadata": metadata,
        "timestamps_sec": list(timestamps_sec[:used_count]),
        "frame_indices": frame_indices[:used_count],
        "frames": frames[:used_count],
    }
