from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


@dataclass
class QAPair:
    question: str
    choices: list[str]
    correct_answer: str
    segment_index: int
    qa_type: str  # "descriptive" or "causal"


@dataclass
class Segment:
    index: int  # 1-indexed
    video_path: str = ""
    start_time: float | None = None
    end_time: float | None = None


@dataclass
class VideoInstance:
    video_id: str
    segments: list[Segment]
    descriptive_qas: dict[int, QAPair]  # segment_index -> QAPair
    causal_qas: dict[int, QAPair]  # segment_index -> QAPair (keys 2..N)

    @property
    def num_segments(self) -> int:
        return len(self.segments)


@dataclass
class ChainEvent:
    segment_index: int
    question_type: str  # "descriptive" or "causal"
    question: str
    model_answer: str
    correct_answer: str
    is_correct: bool
    chain_length_at_step: int
    score_awarded: int


@dataclass
class VideoResult:
    video_id: str
    events: list[ChainEvent]
    total_score: int
    max_chain_length: int
    num_restarts: int
    chain_lengths: list[int]  # length of each completed chain segment
    num_segments: int


@dataclass
class AggregateResults:
    video_results: list[VideoResult]
    csr: float
    amcl: float
    mcl: int
    rf: float
    ws: float
    dua: float
    icra: float

    def to_dict(self) -> dict:
        return {
            "metrics": {
                "csr": self.csr,
                "amcl": self.amcl,
                "mcl": self.mcl,
                "rf": self.rf,
                "ws": self.ws,
                "dua": self.dua,
                "icra": self.icra,
            },
            "per_video": [
                {
                    "video_id": vr.video_id,
                    "total_score": vr.total_score,
                    "max_chain_length": vr.max_chain_length,
                    "num_restarts": vr.num_restarts,
                    "chain_lengths": vr.chain_lengths,
                    "num_segments": vr.num_segments,
                    "events": [asdict(e) for e in vr.events],
                }
                for vr in self.video_results
            ],
        }


def load_dataset(path: str) -> list[VideoInstance]:
    with open(path) as f:
        data = json.load(f)

    videos: list[VideoInstance] = []
    for v in data["videos"]:
        video_id = v["video_id"]

        segments = [
            Segment(
                index=s["index"],
                video_path=s.get("video_path", ""),
                start_time=s.get("start_time"),
                end_time=s.get("end_time"),
            )
            for s in sorted(v["segments"], key=lambda s: s["index"])
        ]

        if not segments:
            raise ValueError(f"Video {video_id}: no segments found")

        expected_indices = list(range(1, len(segments) + 1))
        actual_indices = [s.index for s in segments]
        if actual_indices != expected_indices:
            raise ValueError(
                f"Video {video_id}: segment indices must be contiguous from 1, "
                f"got {actual_indices}"
            )

        n = len(segments)

        descriptive_qas: dict[int, QAPair] = {}
        for qa in v["descriptive_qas"]:
            idx = qa["segment_index"]
            if idx < 1 or idx > n:
                raise ValueError(
                    f"Video {video_id}: descriptive QA segment_index {idx} "
                    f"out of range [1, {n}]"
                )
            descriptive_qas[idx] = QAPair(
                question=qa["question"],
                choices=qa["choices"],
                correct_answer=qa["correct_answer"],
                segment_index=idx,
                qa_type="descriptive",
            )

        for i in range(1, n + 1):
            if i not in descriptive_qas:
                raise ValueError(
                    f"Video {video_id}: missing descriptive QA for segment {i}"
                )

        causal_qas: dict[int, QAPair] = {}
        for qa in v.get("causal_qas", []):
            idx = qa["segment_index"]
            if idx < 2 or idx > n:
                raise ValueError(
                    f"Video {video_id}: causal QA segment_index {idx} "
                    f"out of range [2, {n}]"
                )
            causal_qas[idx] = QAPair(
                question=qa["question"],
                choices=qa["choices"],
                correct_answer=qa["correct_answer"],
                segment_index=idx,
                qa_type="causal",
            )

        for i in range(2, n + 1):
            if i not in causal_qas:
                raise ValueError(
                    f"Video {video_id}: missing causal QA for segment {i}"
                )

        videos.append(
            VideoInstance(
                video_id=video_id,
                segments=segments,
                descriptive_qas=descriptive_qas,
                causal_qas=causal_qas,
            )
        )

    return videos
