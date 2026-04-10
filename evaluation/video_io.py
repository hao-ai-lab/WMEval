from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .context import EvaluationContext, SampledVideoFrames, VideoMetadata

SAMPLE_MODE_TIMESTAMP = "timestamp"
SAMPLE_MODE_FRAME_STEP = "frame_step"


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


def build_default_event_offsets(before: int, after: int) -> List[int]:
    """Integer frame offsets around an event, e.g. before=2, after=2 → [-2, -1, 0, 1, 2]."""
    b = int(before)
    a = int(after)
    return list(range(-b, a + 1))


def build_vlm_keyframe_indices(frame_count: int, max_grid_frames: int) -> List[int]:
    """Evenly spaced frame indices (inclusive of first and last) for VLM grids."""
    fc = int(frame_count)
    if fc <= 0:
        return []
    m = min(int(max_grid_frames), fc)
    if m <= 1:
        return [0]
    raw = np.linspace(0, fc - 1, num=m)
    out: List[int] = []
    seen: set[int] = set()
    for x in raw:
        idx = int(round(float(x)))
        idx = min(max(idx, 0), fc - 1)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


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


def build_frame_step_indices(
    metadata: VideoMetadata,
    target_fps: float,
    max_pairs: int,
) -> Tuple[List[int], List[float]]:

    if metadata.frame_count <= 0:
        return [], []
    fps = metadata.fps if metadata.fps > 0 else 1.0
    step = max(1, int(round(fps / target_fps)))
    indices = list(range(0, metadata.frame_count, step))
    if len(indices) > max_pairs:
        step2 = max(1, len(indices) // max_pairs)
        indices = indices[::step2][:max_pairs]
    timestamps_sec = [i / fps for i in indices]
    return indices, timestamps_sec


def sample_video_frames_by_indices(
    video_path: str,
    frame_indices: Sequence[int],
    metadata: VideoMetadata,
    timestamps_sec: Sequence[float],
) -> SampledVideoFrames:
    frames = load_frames_by_indices(video_path, frame_indices)
    used = min(len(frames), len(frame_indices), len(timestamps_sec))
    return SampledVideoFrames(
        metadata=metadata,
        timestamps_sec=list(timestamps_sec[:used]),
        frame_indices=list(frame_indices[:used]),
        frames=frames[:used],
    )


def build_evaluation_context(
    reference_video: str,
    candidate_video: str,
    eval_fps: float = 4.0,
    max_pairs: int = 32,
    alignment_mode: str = "shared_duration",
    text_prompt: str = "",
    sample_mode: str = SAMPLE_MODE_TIMESTAMP,
) -> EvaluationContext:
    if alignment_mode != "shared_duration":
        raise ValueError(
            f"Unsupported alignment_mode={alignment_mode!r}. "
            "Only 'shared_duration' is currently implemented."
        )
    if sample_mode not in (SAMPLE_MODE_TIMESTAMP, SAMPLE_MODE_FRAME_STEP):
        raise ValueError(
            f"Unsupported sample_mode={sample_mode!r}. "
            f"Use {SAMPLE_MODE_TIMESTAMP!r} or {SAMPLE_MODE_FRAME_STEP!r}."
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

    if sample_mode == SAMPLE_MODE_TIMESTAMP:
        timestamps_sec = build_sample_timestamps(
            duration_sec=shared_duration_sec,
            eval_fps=eval_fps,
            max_pairs=max_pairs,
        )
        reference_sample = sample_video_frames(reference_video, timestamps_sec)
        candidate_sample = sample_video_frames(candidate_video, timestamps_sec)
    else:
        ref_indices, ref_ts = build_frame_step_indices(
            reference_metadata, eval_fps, max_pairs
        )
        cand_indices, cand_ts = build_frame_step_indices(
            candidate_metadata, eval_fps, max_pairs
        )
        n = min(len(ref_indices), len(cand_indices))
        if n == 0:
            raise RuntimeError("frame_step sampling produced no frames")
        reference_sample = sample_video_frames_by_indices(
            reference_video, ref_indices[:n], reference_metadata, ref_ts[:n]
        )
        candidate_sample = sample_video_frames_by_indices(
            candidate_video, cand_indices[:n], candidate_metadata, cand_ts[:n]
        )

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
