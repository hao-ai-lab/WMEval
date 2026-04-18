from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAUSALSTEP_ROOT = ROOT.parent / "CausalStep"
if str(CAUSALSTEP_ROOT) not in sys.path:
    sys.path.insert(0, str(CAUSALSTEP_ROOT))

from evaluation.modules._causal_step_judge import JudgeRequest
from evaluation.modules._causal_step_openai_judge import (
    OpenAIJudge,
    build_prompt,
    sample_segment_frames,
)


def _fake_frames(n: int = 60) -> np.ndarray:
    return (np.random.default_rng(0).random((n, 64, 64, 3)) * 255).astype("uint8")


def test_sample_segment_frames_picks_four_evenly():
    frames = _fake_frames(60)
    picks = sample_segment_frames(frames, start_frame=10, end_frame=50, num_samples=4)
    assert picks.shape == (4, 64, 64, 3)


def test_build_prompt_contains_question_and_choices():
    req = JudgeRequest(
        video_path="x.mp4",
        segment_index=1,
        qa_type="causal",
        question="Why does the beam fall?",
        choices=["A) ball hits it", "B) wind", "C) magic"],
        start_sec=0.5,
        end_sec=1.2,
        segment_frames_start=15,
        segment_frames_end=36,
    )
    prompt = build_prompt(req)
    assert "Why does the beam fall?" in prompt
    assert "A) ball hits it" in prompt
    assert "B) wind" in prompt
    assert "Answer with exactly one letter" in prompt


def test_openai_judge_caches_and_parses(tmp_path: Path):
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="The answer is B."))]
    )

    judge = OpenAIJudge(
        client=fake_client,
        model="gpt-4o",
        cache_dir=tmp_path / "cache",
        frames_provider=lambda path: _fake_frames(60),
    )

    req = JudgeRequest(
        video_path="x.mp4",
        segment_index=1,
        qa_type="causal",
        question="Why does the beam fall?",
        choices=["A) ball hits it", "B) wind", "C) magic"],
        start_sec=0.5,
        end_sec=1.2,
        segment_frames_start=15,
        segment_frames_end=36,
    )
    out = judge.answer(req)
    assert out == "B"
    assert fake_client.chat.completions.create.call_count == 1

    # Second call should hit cache, not the API
    out2 = judge.answer(req)
    assert out2 == "B"
    assert fake_client.chat.completions.create.call_count == 1
