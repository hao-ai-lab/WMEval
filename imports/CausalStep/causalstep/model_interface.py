from __future__ import annotations

import random
from abc import ABC, abstractmethod

from causalstep.schema import Segment


class VideoQAModel(ABC):
    @abstractmethod
    def answer_descriptive(
        self,
        question: str,
        choices: list[str],
        segment: Segment,
    ) -> str:
        """Answer a descriptive QA given a single segment. Returns choice letter."""
        ...

    @abstractmethod
    def answer_causal(
        self,
        question: str,
        choices: list[str],
        prev_answer: str,
        segments: tuple[Segment, Segment],
    ) -> str:
        """Answer a causal QA given two consecutive segments and the prior answer.
        Returns choice letter."""
        ...


class FixedAnswerModel(VideoQAModel):
    """Always returns a fixed answer letter. Useful for deterministic testing."""

    def __init__(self, answer: str = "A"):
        self.answer = answer

    def answer_descriptive(self, question, choices, segment):
        return self.answer

    def answer_causal(self, question, choices, prev_answer, segments):
        return self.answer


class RandomModel(VideoQAModel):
    """Picks a random choice letter. Useful for baseline evaluation."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def _pick(self, choices: list[str]) -> str:
        choice = self.rng.choice(choices)
        return choice.split(")")[0].strip() if ")" in choice else choice[0]

    def answer_descriptive(self, question, choices, segment):
        return self._pick(choices)

    def answer_causal(self, question, choices, prev_answer, segments):
        return self._pick(choices)


class PrecomputedModel(VideoQAModel):
    """Looks up answers from a pre-computed dictionary.

    The dictionary should map (video_id, segment_index, qa_type) -> answer letter.
    This is the primary use case: run inference separately, then evaluate.
    """

    def __init__(
        self,
        answers: dict[tuple[str, int, str], str],
        video_id: str = "",
    ):
        self.answers = answers
        self.video_id = video_id

    def set_video_id(self, video_id: str) -> None:
        self.video_id = video_id

    def answer_descriptive(self, question, choices, segment):
        key = (self.video_id, segment.index, "descriptive")
        if key not in self.answers:
            raise KeyError(f"No precomputed answer for {key}")
        return self.answers[key]

    def answer_causal(self, question, choices, prev_answer, segments):
        key = (self.video_id, segments[1].index, "causal")
        if key not in self.answers:
            raise KeyError(f"No precomputed answer for {key}")
        return self.answers[key]


def load_precomputed_answers(path: str) -> dict[tuple[str, int, str], str]:
    """Load precomputed answers from a JSON file.

    Expected format:
    {
        "answers": [
            {"video_id": "v1", "segment_index": 1, "qa_type": "descriptive", "answer": "A"},
            {"video_id": "v1", "segment_index": 2, "qa_type": "causal", "answer": "B"},
            ...
        ]
    }
    """
    import json

    with open(path) as f:
        data = json.load(f)

    answers: dict[tuple[str, int, str], str] = {}
    for entry in data["answers"]:
        key = (entry["video_id"], entry["segment_index"], entry["qa_type"])
        answers[key] = entry["answer"]
    return answers
