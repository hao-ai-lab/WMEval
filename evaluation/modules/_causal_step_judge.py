"""VLM judge abstraction for the causal_step module.

The real implementation (GPT-4o) is added in a later task. This file defines:
  - JudgeRequest: the inputs the module hands to a judge per QA.
  - Judge Protocol (structurally typed).
  - FakeJudge: deterministic test double returning programmed answers.
  - parse_letter: best-effort extraction of A/B/C/... from model output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Protocol, Tuple


class JudgeParseError(ValueError):
    pass


@dataclass
class JudgeRequest:
    video_path: str
    segment_index: int
    qa_type: str  # "descriptive" or "causal"
    question: str
    choices: List[str]
    start_sec: float
    end_sec: float
    # For causal QAs we may also want the previous segment window; kept optional:
    prev_segment_index: int | None = None
    prev_start_sec: float | None = None
    prev_end_sec: float | None = None
    prev_answer: str | None = None


class Judge(Protocol):
    def answer(self, request: JudgeRequest) -> str: ...


class FakeJudge:
    """Deterministic judge for tests.

    `answers` maps (video_path, segment_index, qa_type) -> answer letter.
    Any request not in the map returns `default`.
    """

    def __init__(
        self,
        answers: Mapping[Tuple[str, int, str], str] | None = None,
        default: str = "A",
    ):
        self._answers: Dict[Tuple[str, int, str], str] = dict(answers or {})
        self._default = default
        self.calls: List[JudgeRequest] = []

    def answer(self, request: JudgeRequest) -> str:
        self.calls.append(request)
        key = (request.video_path, request.segment_index, request.qa_type)
        return self._answers.get(key, self._default)


_LETTER_RE = re.compile(r"(?i)(?:^|[^a-z])([a-z])(?:[^a-z]|$)")


def parse_letter(raw: str, *, choices: List[str]) -> str:
    """Extract the first letter from `raw` that corresponds to one of `choices`.

    `choices` entries look like ``"A) text"`` — we read the leading letter of
    each choice to build the valid set.
    """
    valid = set()
    for c in choices:
        c = c.strip()
        if not c:
            continue
        first = c[0]
        if first.isalpha():
            valid.add(first.upper())
    if not valid:
        raise JudgeParseError(f"no valid choice letters in {choices}")
    for m in _LETTER_RE.finditer(raw):
        letter = m.group(1).upper()
        if letter in valid:
            return letter
    raise JudgeParseError(f"no valid choice letter found in model output: {raw!r}")
