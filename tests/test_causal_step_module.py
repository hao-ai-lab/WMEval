from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAUSALSTEP_ROOT = ROOT.parent / "CausalStep"
if str(CAUSALSTEP_ROOT) not in sys.path:
    sys.path.insert(0, str(CAUSALSTEP_ROOT))

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "beamdomino"


def test_fake_judge_returns_programmed_answers():
    from evaluation.modules._causal_step_judge import FakeJudge, JudgeRequest

    judge = FakeJudge(
        answers={
            ("unity_gt.mp4", 1, "descriptive"): "A",
            ("unity_gt.mp4", 2, "causal"): "B",
        },
        default="Z",
    )
    req1 = JudgeRequest(
        video_path="unity_gt.mp4",
        segment_index=1,
        qa_type="descriptive",
        question="q",
        choices=["A) x", "B) y"],
        start_sec=0.0,
        end_sec=0.5,
    )
    assert judge.answer(req1) == "A"

    req2 = JudgeRequest(
        video_path="unity_gt.mp4",
        segment_index=2,
        qa_type="causal",
        question="q",
        choices=["A) x", "B) y"],
        start_sec=0.5,
        end_sec=1.0,
    )
    assert judge.answer(req2) == "B"

    req3 = JudgeRequest(
        video_path="unity_gt.mp4",
        segment_index=3,
        qa_type="descriptive",
        question="q",
        choices=["A) x"],
        start_sec=1.0,
        end_sec=1.5,
    )
    assert judge.answer(req3) == "Z"


def test_parse_letter_extracts_first_choice_letter():
    from evaluation.modules._causal_step_judge import parse_letter

    assert parse_letter("A", choices=["A) x", "B) y"]) == "A"
    assert parse_letter("The answer is B.", choices=["A) x", "B) y"]) == "B"
    assert parse_letter("(c)", choices=["A) x", "B) y", "C) z"]) == "C"


def test_parse_letter_raises_on_no_match():
    import pytest
    from evaluation.modules._causal_step_judge import parse_letter, JudgeParseError

    with pytest.raises(JudgeParseError):
        parse_letter("I do not know.", choices=["A) x", "B) y"])
