from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

from .context import EvaluationContext, SampledVideoFrames, VideoMetadata


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
) -> SampledVideoFrames:
    metadata = get_video_metadata(video_path)
    frame_indices = [timestamp_to_frame_index(t, metadata) for t in timestamps_sec]
    frames = load_frames_by_indices(video_path, frame_indices)

    used_count = min(len(frames), len(frame_indices), len(timestamps_sec))
    return SampledVideoFrames(
        metadata=metadata,
        timestamps_sec=list(timestamps_sec[:used_count]),
        frame_indices=frame_indices[:used_count],
        frames=frames[:used_count],
    )


def build_evaluation_context(
    reference_video: str,
    candidate_video: str,
    eval_fps: float = 4.0,
    max_pairs: int = 32,
    alignment_mode: str = "shared_duration",
    text_prompt: str = "",
) -> EvaluationContext:
    if alignment_mode != "shared_duration":
        raise ValueError(
            f"Unsupported alignment_mode={alignment_mode!r}. "
            "Only 'shared_duration' is currently implemented."
        )

    reference_metadata = get_video_metadata(reference_video)
    candidate_metadata = get_video_metadata(candidate_video)

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

    reference_sample = sample_video_frames(reference_video, timestamps_sec)
    candidate_sample = sample_video_frames(candidate_video, timestamps_sec)

    return EvaluationContext(
        reference_video=reference_video,
        candidate_video=candidate_video,
        reference=reference_sample,
        candidate=candidate_sample,
        shared_duration_sec=shared_duration_sec,
        eval_fps=eval_fps,
        max_pairs=max_pairs,
        alignment_mode=alignment_mode,
        text_prompt=text_prompt,
    )
