from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evaluation.lite.adapters.vlm_consistency_adapter import VlmConsistencyAdapter
from evaluation.lite.case_loader import load_case
from evaluation.lite.candidate_discovery import CandidateRow
from evaluation.lite.suite_loader import load_suite
from lite_suite_test_utils import (
    RUBRIC_SCENE_NAME,
    FakeJudgeResult,
    build_default_suite_dir,
    write_rubric,
)


def _setup(tmp_path: Path):
    suite_root = build_default_suite_dir(tmp_path / "suite")
    suite = load_suite(suite_root)
    dimension = suite.dimensions["toy_promptcons"]
    case = load_case(
        suite.cases_root / "scene_b/first", allowed_cases=suite.all_case_ids()
    )
    video = tmp_path / "candidate.mp4"
    video.write_text("fake\n", encoding="utf-8")
    candidate = CandidateRow("genie3", "scene_b/first", "0001", video)
    return suite, dimension, case, candidate, video


def test_vlm_adapter_hits_pr17_rubric_naming(tmp_path: Path) -> None:
    suite, dimension, case, candidate, video = _setup(tmp_path)
    rubric_path = write_rubric(suite.root)
    # PR #17's committed layout: rubrics/<OriginalSceneName>/<provider>_<model>_<version>.json
    assert rubric_path == (
        suite.root / "rubrics" / RUBRIC_SCENE_NAME / "openai_gpt-5-4_v1.json"
    )

    judge_calls: list[dict[str, Any]] = []

    def fake_judge(**kwargs: Any) -> FakeJudgeResult:
        judge_calls.append(kwargs)
        return FakeJudgeResult(0.9, 1.0, 0.75)

    frame_calls: list[dict[str, Any]] = []

    def fake_frames(**kwargs: Any):
        frame_calls.append(kwargs)
        return [("ref_frame", "b64"), ("frame_t=0.0s", "b64")], 1

    adapter = VlmConsistencyAdapter(judge=fake_judge, frame_preparer=fake_frames)
    result = adapter.evaluate(
        suite=suite,
        dimension=dimension,
        case=case,
        candidate=candidate,
        candidate_video=video,
    )

    assert result.status == "completed"
    assert result.metrics == {
        "overall_score": 0.9,
        "env_score": 1.0,
        "obj_score": 0.75,
    }
    assert result.full_report["overall_score"] == 0.9

    # rubric resolved via case.rubric_scene_name, judged with suite vlm config
    call = judge_calls[0]
    assert call["rubric"].scene_name == RUBRIC_SCENE_NAME
    assert call["provider"] == "openai"
    assert call["model"] == "gpt-5.4"
    assert call["alpha"] == 0.6

    # frame sampling mirrors batch.py: 10s window, 18 frames, ref first frame
    frames_kwargs = frame_calls[0]
    assert frames_kwargs["video_path"] == str(video)
    assert frames_kwargs["first_n_seconds"] == 10.0
    assert frames_kwargs["max_frames"] == 18
    assert frames_kwargs["gt_frame_path"] == str(case.first_frame)
    assert frames_kwargs["use_timestamp_labels"] is True


def test_vlm_adapter_missing_rubric_status(tmp_path: Path) -> None:
    suite, dimension, case, candidate, video = _setup(tmp_path)
    # no rubric written

    adapter = VlmConsistencyAdapter(
        judge=lambda **kwargs: FakeJudgeResult(1.0, 1.0, 1.0),
        frame_preparer=lambda **kwargs: ([], 0),
    )
    result = adapter.evaluate(
        suite=suite,
        dimension=dimension,
        case=case,
        candidate=candidate,
        candidate_video=video,
    )

    assert result.status == "missing_rubric"
    assert "generate-rubrics" in (result.error or "")


def test_vlm_adapter_skips_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite, dimension, case, candidate, video = _setup(tmp_path)
    write_rubric(suite.root)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    adapter = VlmConsistencyAdapter()  # real judge would need a key
    result = adapter.evaluate(
        suite=suite,
        dimension=dimension,
        case=case,
        candidate=candidate,
        candidate_video=video,
    )

    assert result.status == "skipped_no_api_key"
    assert "OPENAI" in (result.error or "").upper() or "openai" in (result.error or "")
