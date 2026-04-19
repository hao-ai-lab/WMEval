from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAUSALSTEP_ROOT = ROOT / "imports" / "CausalStep"
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


def test_fake_judge_selection_emits_warning(tmp_path, caplog):
    """fake_always_a is a debug judge; using it silently is a footgun because the
    BeamDomino templates all have correct_answer 'A' so metrics look perfect."""
    import json
    import logging
    import numpy as np
    from evaluation.modules.causal_step import analyze_causal_step_from_context
    from evaluation.context import EvaluationContext, SampledVideoFrames, VideoMetadata

    qa_path = tmp_path / "qa.json"
    qa_path.write_text(json.dumps({
        "videos": [{
            "video_id": "vid1",
            "segments": [
                {"index": 1, "start_time": 0.0, "end_time": 0.33},
                {"index": 2, "start_time": 0.33, "end_time": 0.66},
            ],
            "descriptive_qas": [
                {"segment_index": 1, "question": "Q1?", "choices": ["A) yes", "B) no"], "correct_answer": "A"},
                {"segment_index": 2, "question": "Q2?", "choices": ["A) yes", "B) no"], "correct_answer": "A"},
            ],
            "causal_qas": [
                {"segment_index": 2, "question": "C2?", "choices": ["A) cause", "B) other"], "correct_answer": "A"},
            ],
        }]
    }))

    def _stub_frames() -> SampledVideoFrames:
        meta = VideoMetadata(path="stub.mp4", fps=30.0, frame_count=10, width=64, height=64, duration_sec=0.33)
        arr = np.zeros((64, 64, 3), dtype="uint8")
        return SampledVideoFrames(metadata=meta, timestamps_sec=[0.0], frame_indices=[0], frames=[arr])

    ctx = EvaluationContext(
        reference_video=str(tmp_path / "unity.mp4"),
        candidate_video=str(tmp_path / "wm.mp4"),
        reference=_stub_frames(),
        candidate=_stub_frames(),
        shared_duration_sec=0.66,
        eval_fps=30.0,
        max_pairs=1,
        artifacts={
            "causal_step_qa_dataset": str(qa_path),
            "causal_step_video_id": "vid1",
            "causal_step_answers_cache_dir": str(tmp_path / "cache"),
            "causal_step_causal_fail_advance": True,
            "causal_step_judge": "fake_always_a",
        },
    )
    with caplog.at_level(logging.WARNING, logger="evaluation.modules.causal_step"):
        out = analyze_causal_step_from_context(context=ctx)
    assert any("fake_always_a" in rec.message for rec in caplog.records), (
        f"no fake_always_a warning emitted; got: {[r.message for r in caplog.records]}"
    )
    # Judge kind is also surfaced in the result so downstream consumers can filter.
    assert out.get("judge_kind") == "fake_always_a"


def test_analyze_causal_step_from_context_uses_fake_judge(tmp_path):
    import json
    import numpy as np
    from evaluation.modules.causal_step import analyze_causal_step_from_context
    from evaluation.context import EvaluationContext, SampledVideoFrames, VideoMetadata

    qa_path = tmp_path / "qa.json"
    qa_path.write_text(json.dumps({
        "videos": [{
            "video_id": "vid1",
            "segments": [
                {"index": 1, "start_time": 0.0, "end_time": 0.33},
                {"index": 2, "start_time": 0.33, "end_time": 0.66},
            ],
            "descriptive_qas": [
                {"segment_index": 1, "question": "Q1?", "choices": ["A) yes", "B) no"], "correct_answer": "A"},
                {"segment_index": 2, "question": "Q2?", "choices": ["A) yes", "B) no"], "correct_answer": "A"},
            ],
            "causal_qas": [
                {"segment_index": 2, "question": "C2?", "choices": ["A) cause", "B) other"], "correct_answer": "A"},
            ],
        }]
    }))

    def _stub_frames() -> SampledVideoFrames:
        meta = VideoMetadata(path="stub.mp4", fps=30.0, frame_count=10, width=64, height=64, duration_sec=0.33)
        arr = np.zeros((64, 64, 3), dtype="uint8")
        return SampledVideoFrames(metadata=meta, timestamps_sec=[0.0], frame_indices=[0], frames=[arr])

    ctx = EvaluationContext(
        reference_video=str(tmp_path / "unity.mp4"),
        candidate_video=str(tmp_path / "wm.mp4"),
        reference=_stub_frames(),
        candidate=_stub_frames(),
        shared_duration_sec=0.66,
        eval_fps=30.0,
        max_pairs=1,
        artifacts={
            "causal_step_qa_dataset": str(qa_path),
            "causal_step_video_id": "vid1",
            "causal_step_answers_cache_dir": str(tmp_path / "cache"),
            "causal_step_causal_fail_advance": True,
            "causal_step_judge": "fake_always_a",
        },
    )
    out = analyze_causal_step_from_context(context=ctx)
    assert out["module"] == "causal_step"
    assert "unity" in out and "wm" in out and "delta" in out
    assert "CSR" in out["unity"]
    assert out["unity"]["CSR"] == 100.0
    # delta keys are lowercase per compute_delta convention
    assert "csr" in out["delta"]
