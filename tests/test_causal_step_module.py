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
    # All three blocks (unity, wm, delta) use lowercase metric keys —
    # matching causalstep.metrics.compute_all_metrics output.
    assert "csr" in out["unity"]
    assert out["unity"]["csr"] == 100.0
    assert "csr" in out["delta"]


def _causal_ctx(tmp_path, *, artifacts_extra: dict):
    """EvaluationContext over a 2-segment dataset, live judge = always 'A'."""
    import json
    import numpy as np
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

    artifacts = {
        "causal_step_qa_dataset": str(qa_path),
        "causal_step_video_id": "vid1",
        "causal_step_answers_cache_dir": str(tmp_path / "cache"),
        "causal_step_causal_fail_advance": True,
        "causal_step_judge": "fake_always_a",
    }
    artifacts.update(artifacts_extra)
    return EvaluationContext(
        reference_video=str(tmp_path / "unity.mp4"),
        candidate_video=str(tmp_path / "wm.mp4"),
        reference=_stub_frames(),
        candidate=_stub_frames(),
        shared_duration_sec=0.66,
        eval_fps=30.0,
        max_pairs=1,
        artifacts=artifacts,
    )


def _write_recording(path, passes: list[dict[str, str]], video_id: str = "vid1"):
    import json

    path.write_text(json.dumps({
        "video_id": video_id,
        "judge": {"model": "gpt-5.4", "num_frame_samples": 8,
                  "prompt_version": "v1", "temperature": 0.0},
        "passes": [
            {
                "video_id": video_id,
                "answers": [
                    {"video_id": video_id,
                     "segment_index": int(k.split("|")[0]),
                     "qa_type": k.split("|")[1],
                     "answer": v}
                    for k, v in p.items()
                ],
            }
            for p in passes
        ],
    }))
    return path


def test_recorded_reference_answers_replace_the_live_judge_on_the_reference_side(tmp_path):
    """The reference side must replay the recording, not re-judge the video.

    The live judge here answers 'A' to everything, which would complete the
    chain. The recording deliberately disagrees on segment 1, so a unity CSR
    of 0 alongside a wm CSR of 100 is only reachable if the reference read the
    recording while the candidate still ran the live judge.
    """
    from evaluation.modules.causal_step import analyze_causal_step_from_context

    rec = _write_recording(
        tmp_path / "judge_answers.json",
        [{"1|descriptive": "B", "2|descriptive": "A", "2|causal": "A"}],
    )
    ctx = _causal_ctx(tmp_path, artifacts_extra={"causal_step_reference_answers": str(rec)})

    out = analyze_causal_step_from_context(context=ctx)

    assert out["unity"]["csr"] == 0.0
    assert out["wm"]["csr"] == 100.0


def test_recorded_reference_answers_reproduce_a_fractional_unity_csr(tmp_path):
    """Five recorded passes, three completing the chain, give Unity CSR 60.

    This is the paper's per-shape reference figure on `glancing`, which a
    single pass cannot express: CSR is all-or-nothing per video.
    """
    from evaluation.modules.causal_step import analyze_causal_step_from_context

    good = {"1|descriptive": "A", "2|descriptive": "A", "2|causal": "A"}
    bad = {"1|descriptive": "B", "2|descriptive": "A", "2|causal": "A"}
    rec = _write_recording(tmp_path / "judge_answers.json", [good, good, good, bad, bad])
    ctx = _causal_ctx(
        tmp_path,
        artifacts_extra={
            "causal_step_reference_answers": str(rec),
            "causal_step_judge_repeats": 5,
        },
    )

    out = analyze_causal_step_from_context(context=ctx)

    assert out["unity"]["csr"] == 60.0
    assert out["reference_answers_passes"] == 5


def test_repeats_default_to_the_recorded_pass_count(tmp_path):
    """A 5-pass recording must not be silently scored off pass 0 alone.

    The recording is the benchmark's reference fixture; how many passes it
    holds is a property of the fixture, so it sets the default repeat count
    rather than requiring every caller to restate it.
    """
    from evaluation.modules.causal_step import analyze_causal_step_from_context

    good = {"1|descriptive": "A", "2|descriptive": "A", "2|causal": "A"}
    bad = {"1|descriptive": "B", "2|descriptive": "A", "2|causal": "A"}
    rec = _write_recording(tmp_path / "judge_answers.json", [good, good, good, bad, bad])
    ctx = _causal_ctx(tmp_path, artifacts_extra={"causal_step_reference_answers": str(rec)})

    out = analyze_causal_step_from_context(context=ctx)

    assert out["unity"]["csr"] == 60.0
    assert out["judge_repeats"] == 5


def test_wm_full_video_defaults_to_the_published_protocol(tmp_path):
    """App. C.5 scores the candidate over its whole clip while the reference
    stays segment-windowed, because world-model rollouts are not frame-aligned
    to Unity. That asymmetry is the published protocol, so it is the default."""
    from evaluation.modules.causal_step import analyze_causal_step_from_context

    (tmp_path / "unity.mp4").write_bytes(b"unity-bytes")
    (tmp_path / "wm.mp4").write_bytes(b"different-bytes")
    ctx = _causal_ctx(tmp_path, artifacts_extra={})

    out = analyze_causal_step_from_context(context=ctx)

    assert out["wm_full_video"] is True


def test_identical_candidate_forces_segment_windowing(tmp_path):
    """A candidate that IS the reference must measure zero.

    Full-video windowing asks the candidate a different question than the
    reference, which on a byte-identical pair reported csr_delta = 100 —
    maximal apparent world-model failure from the same file. Identity is
    detectable, so the module refuses the asymmetry instead of reporting it.
    """
    from evaluation.modules.causal_step import analyze_causal_step_from_context

    same = b"identical-video-bytes"
    (tmp_path / "unity.mp4").write_bytes(same)
    (tmp_path / "wm.mp4").write_bytes(same)
    ctx = _causal_ctx(tmp_path, artifacts_extra={"causal_step_wm_full_video": True})

    out = analyze_causal_step_from_context(context=ctx)

    assert out["wm_full_video"] is False
    assert "identical" in out["wm_full_video_reason"]


class _SettingsJudge:
    """Always-'A' judge that reports live settings, like `OpenAIJudge`."""

    def __init__(self, **settings):
        self._settings = {"model": "gpt-5.4", "num_frame_samples": 8,
                          "prompt_version": "v1", "temperature": None, **settings}

    def answer(self, request):
        return "A"

    def describe_settings(self):
        return dict(self._settings)


def test_recording_from_a_different_judge_config_is_refused(tmp_path, monkeypatch):
    """The reference replays a K=8 recording; a candidate judged at K=4 must
    not be silently paired with it."""
    import pytest
    from evaluation.modules import causal_step
    from evaluation.modules._causal_step_recorded_judge import RecordedAnswersError

    rec = _write_recording(
        tmp_path / "judge_answers.json",
        [{"1|descriptive": "A", "2|descriptive": "A", "2|causal": "A"}],
    )
    ctx = _causal_ctx(tmp_path, artifacts_extra={"causal_step_reference_answers": str(rec)})
    monkeypatch.setattr(causal_step, "_build_judge", lambda context: _SettingsJudge(num_frame_samples=4))

    with pytest.raises(RecordedAnswersError, match="num_frame_samples"):
        causal_step.analyze_causal_step_from_context(context=ctx)


def test_recording_from_the_same_judge_config_is_replayed(tmp_path, monkeypatch):
    from evaluation.modules import causal_step

    rec = _write_recording(
        tmp_path / "judge_answers.json",
        [{"1|descriptive": "B", "2|descriptive": "A", "2|causal": "A"}],
    )
    ctx = _causal_ctx(tmp_path, artifacts_extra={"causal_step_reference_answers": str(rec)})
    monkeypatch.setattr(causal_step, "_build_judge", lambda context: _SettingsJudge())

    out = causal_step.analyze_causal_step_from_context(context=ctx)

    assert out["unity"]["csr"] == 0.0
    assert out["wm"]["csr"] == 100.0
