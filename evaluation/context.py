from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

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


@dataclass
class SampledVideoFrames:
    metadata: VideoMetadata
    timestamps_sec: List[float]
    frame_indices: List[int]
    frames: List[np.ndarray]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "timestamps_sec": list(self.timestamps_sec),
            "frame_indices": list(self.frame_indices),
            "num_frames": len(self.frames),
        }


@dataclass
class EvaluationContext:
    reference_video: str
    candidate_video: str
    reference: SampledVideoFrames
    candidate: SampledVideoFrames
    shared_duration_sec: float
    eval_fps: float
    max_pairs: int
    alignment_mode: str = "shared_duration"
    text_prompt: str = ""
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_pairs(self) -> int:
        return min(len(self.reference.frames), len(self.candidate.frames))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference_video": self.reference_video,
            "candidate_video": self.candidate_video,
            "reference_metadata": self.reference.metadata.to_dict(),
            "candidate_metadata": self.candidate.metadata.to_dict(),
            "shared_duration_sec": self.shared_duration_sec,
            "eval_fps": self.eval_fps,
            "max_pairs": self.max_pairs,
            "alignment_mode": self.alignment_mode,
            "text_prompt": self.text_prompt,
            "num_pairs": self.num_pairs,
            "timestamps_sec": self.reference.timestamps_sec[:self.num_pairs],
            "reference_frame_indices": self.reference.frame_indices[:self.num_pairs],
            "candidate_frame_indices": self.candidate.frame_indices[:self.num_pairs],
        }
