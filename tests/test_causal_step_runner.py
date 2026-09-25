from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAUSALSTEP_ROOT = ROOT / "imports" / "CausalStep"
if str(CAUSALSTEP_ROOT) not in sys.path:
    sys.path.insert(0, str(CAUSALSTEP_ROOT))

from evaluation.modules._causal_step_judge import FakeJudge, JudgeRequest
from evaluation.modules._causal_step_runner import run_causal_step_for_video


class _AlwaysA:
    def __init__(self) -> None:
        self.calls: list[JudgeRequest] = []

    def answer(self, request: JudgeRequest) -> str:
        self.calls.append(request)
        return "A"


def test_run_causal_step_for_video_returns_metrics(tmp_path: Path):
    dataset = {
        "videos": [
            {
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
            }
        ]
    }
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(json.dumps(dataset))

    metrics = run_causal_step_for_video(
        qa_dataset_path=qa_path,
        video_id="vid1",
        video_path="fake.mp4",
        judge=_AlwaysA(),
        answers_out_dir=tmp_path / "answers",
        causal_fail_advance=True,
        fps=30.0,
    )
    assert "csr" in metrics
    assert set(metrics.keys()) >= {"csr", "amcl", "mcl", "rf", "ws", "dua", "icra"}
    # _AlwaysA always returns "A"; since every correct_answer is "A", chain should succeed.
    assert metrics["csr"] == 100.0


def _two_segment_dataset() -> dict:
    return {
        "videos": [
            {
                "video_id": "vid1",
                "segments": [
                    {"index": 1, "start_time": 0.0, "end_time": 0.5},
                    {"index": 2, "start_time": 0.5, "end_time": 1.0},
                ],
                "descriptive_qas": [
                    {"segment_index": 1, "question": "Q1?", "choices": ["A) yes", "B) no"], "correct_answer": "A"},
                    {"segment_index": 2, "question": "Q2?", "choices": ["A) yes", "B) no"], "correct_answer": "A"},
                ],
                "causal_qas": [
                    {"segment_index": 2, "question": "C2?", "choices": ["A) cause", "B) other"], "correct_answer": "A"},
                ],
            }
        ]
    }


def test_runner_populates_segment_frame_indices_from_fps(tmp_path: Path):
    """Regression: frame window must be computed from (fps, start_sec, end_sec).

    Previously the runner left segment_frames_start/end as None, which crashed the
    OpenAIJudge with TypeError on `None <= None`.
    """
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(json.dumps(_two_segment_dataset()))

    judge = _AlwaysA()
    run_causal_step_for_video(
        qa_dataset_path=qa_path,
        video_id="vid1",
        video_path="fake.mp4",
        judge=judge,
        answers_out_dir=tmp_path / "answers",
        causal_fail_advance=True,
        fps=30.0,
    )

    # 3 QAs total: desc(seg=1), desc(seg=2), causal(seg=2)
    assert len(judge.calls) == 3
    for call in judge.calls:
        assert call.segment_frames_start is not None, "segment_frames_start was None"
        assert call.segment_frames_end is not None, "segment_frames_end was None"
        assert call.segment_frames_start < call.segment_frames_end

    # Segment 1: [0.0, 0.5)s at 30fps -> [0, 15)
    seg1 = next(c for c in judge.calls if c.segment_index == 1 and c.qa_type == "descriptive")
    assert seg1.segment_frames_start == 0
    assert seg1.segment_frames_end == 15

    # Segment 2: [0.5, 1.0)s at 30fps -> [15, 30)
    seg2_desc = next(c for c in judge.calls if c.segment_index == 2 and c.qa_type == "descriptive")
    assert seg2_desc.segment_frames_start == 15
    assert seg2_desc.segment_frames_end == 30


def test_runner_requires_fps(tmp_path: Path):
    """fps must be passed explicitly — there is no sensible default."""
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(json.dumps(_two_segment_dataset()))

    with pytest.raises(TypeError):
        run_causal_step_for_video(  # type: ignore[call-arg]
            qa_dataset_path=qa_path,
            video_id="vid1",
            video_path="fake.mp4",
            judge=_AlwaysA(),
            answers_out_dir=tmp_path / "answers",
            causal_fail_advance=True,
        )


class _RepeatVaryingJudge:
    """Answers correctly on `correct_repeats`, wrongly on every other pass.

    Stands in for the real judge's run-to-run disagreement at temperature 0.
    """

    def __init__(self, correct_repeats: set[int]) -> None:
        self._correct = correct_repeats
        self.calls: list[JudgeRequest] = []

    def answer(self, request: JudgeRequest) -> str:
        self.calls.append(request)
        return "A" if request.repeat_index in self._correct else "B"


def test_repeats_produce_fractional_csr(tmp_path: Path):
    """CSR over N repeats is successes/N, matching the paper's cell semantics.

    A single pass can only ever report csr 0 or 100 because CSR is
    all-or-nothing per video. The published per-shape figures (e.g. Unity
    CSR 60 on glancing) are means over five passes of the same video, so the
    runner has to hand N results to compute_all_metrics, not one.
    """
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(json.dumps(_two_segment_dataset()))

    judge = _RepeatVaryingJudge(correct_repeats={0, 1, 2})
    metrics = run_causal_step_for_video(
        qa_dataset_path=qa_path,
        video_id="vid1",
        video_path="fake.mp4",
        judge=judge,
        answers_out_dir=tmp_path / "answers",
        causal_fail_advance=True,
        fps=30.0,
        repeats=5,
    )

    assert metrics["csr"] == pytest.approx(60.0)
    assert {r.repeat_index for r in judge.calls} == {0, 1, 2, 3, 4}


def test_single_repeat_is_the_default_and_matches_previous_behaviour(tmp_path: Path):
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(json.dumps(_two_segment_dataset()))

    judge = _RepeatVaryingJudge(correct_repeats={0})
    metrics = run_causal_step_for_video(
        qa_dataset_path=qa_path,
        video_id="vid1",
        video_path="fake.mp4",
        judge=judge,
        answers_out_dir=tmp_path / "answers",
        causal_fail_advance=True,
        fps=30.0,
    )

    assert metrics["csr"] == pytest.approx(100.0)
    assert {r.repeat_index for r in judge.calls} == {0}
