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
    rubric_path = write_rubric(case.root)
    # rubric ships inside the case bundle, keeping PR #17's
    # <SceneName>/<provider>_<model>_<version>.json cache convention
    assert rubric_path == (
        case.root
        / "reference"
        / "artifacts"
        / "rubrics"
        / RUBRIC_SCENE_NAME
        / "openai_gpt-5-4_v1.json"
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
    write_rubric(case.root)
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


class _ScriptedJudge:
    """Fake judge returning the next scripted (overall, env, obj) per call."""

    def __init__(self, scores: list[tuple[float, float, float]]) -> None:
        self._scores = list(scores)

    def __call__(self, **kwargs: Any) -> FakeJudgeResult:
        return FakeJudgeResult(*self._scores.pop(0))


def _freeze_toy_reference(tmp_path: Path, scores: list[tuple[float, float, float]]):
    from evaluation.lite.validation import validate_case_bundle
    from lite_suite_test_utils import regenerate_checksums
    from tools import freeze_prompt_reference_scores as freezer

    suite, dimension, case, candidate, video = _setup(tmp_path)
    write_rubric(case.root)
    unity = case.root / "reference" / "unity_groundtruth.mp4"
    unity.write_bytes(b"fake unity render\n")
    regenerate_checksums(case.root)

    reference_judge = VlmConsistencyAdapter(
        judge=_ScriptedJudge(scores),
        frame_preparer=lambda **kwargs: ([("frame_t=0.0s", "b64")], 1),
    )
    output = freezer.freeze(
        case.case_id,
        passes=len(scores),
        workers=1,
        suite_root=suite.root,
        dimension_id=dimension.dimension_id,
        adapter=reference_judge,
    )
    freezer.register_in_case(case.root)
    regenerate_checksums(case.root)
    frozen_case = load_case(case.root, allowed_cases=suite.all_case_ids())
    validate_case_bundle(frozen_case)
    return suite, dimension, frozen_case, candidate, video, output


def _candidate_adapter() -> VlmConsistencyAdapter:
    return VlmConsistencyAdapter(
        judge=lambda **kwargs: FakeJudgeResult(0.5, 0.5, 0.5),
        frame_preparer=lambda **kwargs: ([("frame_t=0.0s", "b64")], 1),
    )


def test_frozen_reference_scores_are_read_not_rejudged(tmp_path: Path) -> None:
    import json

    suite, dimension, case, candidate, video, output = _freeze_toy_reference(
        tmp_path, [(0.9, 1.0, 0.75), (0.8, 0.9, 0.65), (1.0, 1.0, 1.0)]
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    assert output.name == "reference_scores.json"
    assert document["reference_video"] == "reference/unity_groundtruth.mp4"
    assert [p["overall_score"] for p in document["passes"]] == [0.9, 0.8, 1.0]
    assert document["summary"]["overall_score"]["mean"] == 0.9
    assert document["summary"]["overall_score"]["n"] == 3
    assert document["judge"]["prompt_version"] == "v1"
    assert document["judge"]["alpha"] == 0.6

    result = _candidate_adapter().evaluate(
        suite=suite,
        dimension=dimension,
        case=case,
        candidate=candidate,
        candidate_video=video,
    )

    assert result.status == "completed"
    assert result.metrics["overall_score"] == 0.5
    assert result.metrics["overall_score_unity"] == 0.9
    assert result.metrics["env_score_unity"] == round((1.0 + 0.9 + 1.0) / 3, 4)
    assert result.metrics["obj_score_unity"] == round((0.75 + 0.65 + 1.0) / 3, 4)
    assert result.full_report["reference_scores"]["status"] == "ok"
    assert result.full_report["reference_scores"]["num_passes"] == 3
    # the live row records the same judge identity the frozen file was made under
    assert result.full_report["judge"] == document["judge"]


def test_frozen_reference_scores_go_stale_when_judge_settings_change(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    suite, dimension, case, candidate, video, _ = _freeze_toy_reference(
        tmp_path, [(0.9, 1.0, 0.75)]
    )
    changed = replace(dimension, vlm={**dimension.vlm, "alpha": 0.5})

    result = _candidate_adapter().evaluate(
        suite=suite,
        dimension=changed,
        case=case,
        candidate=candidate,
        candidate_video=video,
    )

    assert result.status == "completed"
    assert "overall_score_unity" not in result.metrics
    assert result.full_report["reference_scores"]["status"] == "stale"
    assert result.full_report["reference_scores"]["mismatches"] == ["alpha"]


def test_frozen_reference_scores_go_stale_when_unity_video_changes(
    tmp_path: Path,
) -> None:
    suite, dimension, case, candidate, video, _ = _freeze_toy_reference(
        tmp_path, [(0.9, 1.0, 0.75)]
    )
    (case.root / "reference" / "unity_groundtruth.mp4").write_bytes(b"re-rendered\n")

    result = _candidate_adapter().evaluate(
        suite=suite,
        dimension=dimension,
        case=case,
        candidate=candidate,
        candidate_video=video,
    )

    assert "overall_score_unity" not in result.metrics
    assert result.full_report["reference_scores"]["mismatches"] == [
        "reference_video_sha256"
    ]


def test_case_without_frozen_reference_reports_not_frozen(tmp_path: Path) -> None:
    suite, dimension, case, candidate, video = _setup(tmp_path)
    write_rubric(case.root)

    result = _candidate_adapter().evaluate(
        suite=suite,
        dimension=dimension,
        case=case,
        candidate=candidate,
        candidate_video=video,
    )

    assert result.status == "completed"
    assert result.full_report["reference_scores"] == {"status": "not_frozen"}
    assert set(result.metrics) == {"overall_score", "env_score", "obj_score"}
