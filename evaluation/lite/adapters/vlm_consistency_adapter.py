"""Adapter that drives the prompt-consistency VLM judge.

Wraps the existing ``evaluation.modules.vlm_consistency`` pipeline (PR #14 /
PR #17) as a suite dimension WITHOUT touching its internals: rubrics are read
from the suite's committed rubric cache via ``find_rubric`` (missing rubric →
``missing_rubric`` row, never generated inline — generation stays an offline
authoring step), frames are sampled with the exact ``batch.py`` constants, and
scoring goes through ``judge_scene``.

The heavyweight imports (cv2, pydantic, provider clients) happen inside
``evaluate`` so dry runs and suite loading stay dependency-light.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..case_loader import LiteCase
from ..candidate_discovery import CandidateRow
from ..suite_loader import DimensionSpec, Suite
from .base import (
    STATUS_COMPLETED,
    STATUS_MISSING_RUBRIC,
    STATUS_SKIPPED_NO_API_KEY,
    CaseResult,
)


def rubric_scene_name(case: LiteCase) -> str:
    """The scene name rubrics are cached under (PR #17 kept the original Unity
    scene folder names, e.g. ``MirrorRollingBall_first``)."""
    return case.rubric_scene_name or case.scene


class VlmConsistencyAdapter:
    """DimensionEvaluator backed by the vlm_consistency judge pipeline."""

    def __init__(
        self,
        judge: Callable[..., Any] | None = None,
        frame_preparer: Callable[..., Any] | None = None,
    ) -> None:
        self._judge = judge
        self._frame_preparer = frame_preparer

    def evaluate(
        self,
        *,
        suite: Suite,
        dimension: DimensionSpec,
        case: LiteCase,
        candidate: CandidateRow,
        candidate_video: Path,
        device: str | None = None,
    ) -> CaseResult:
        cfg = dimension.vlm
        provider = str(cfg["provider"])
        model = str(cfg["model"])
        prompt_version = str(cfg["prompt_version"])
        alpha = float(cfg["alpha"])
        rubric_dir = suite.root / str(cfg["rubric_dir"])

        from evaluation.modules.vlm_consistency.rubric_cache import find_rubric
        from evaluation.modules.vlm_consistency.schemas import Rubric

        scene_name = rubric_scene_name(case)
        rubric_path = find_rubric(
            scene_name,
            prompt_version,
            str(rubric_dir),
            provider=provider,
            model=model,
        )
        if rubric_path is None:
            return CaseResult(
                status=STATUS_MISSING_RUBRIC,
                error=(
                    f"no cached rubric for scene {scene_name!r} "
                    f"({provider}/{model}/{prompt_version}) under {rubric_dir}; "
                    f"run the generate-rubrics subcommand first"
                ),
            )
        rubric = Rubric.model_validate_json(
            Path(rubric_path).read_text(encoding="utf-8")
        )

        from evaluation.modules.vlm_consistency.vlm_judge import _resolve_api_key

        if self._judge is None:
            try:
                _resolve_api_key(provider)
            except Exception as exc:
                return CaseResult(
                    status=STATUS_SKIPPED_NO_API_KEY,
                    error=f"no API key for provider {provider!r}: {exc}",
                )

        # Sampling parameters are imported from batch.py so suite runs stay in
        # lockstep with the standalone vlm_consistency CLI (PR #17 pipeline).
        from evaluation.modules.vlm_consistency.batch import (
            SAMPLE_DURATION,
            SAMPLE_FRAMES,
        )
        from evaluation.modules.vlm_consistency.frame_sampler import (
            prepare_frames_for_vlm,
        )
        from evaluation.modules.vlm_consistency.vlm_judge import judge_scene

        prepare = self._frame_preparer or prepare_frames_for_vlm
        gt_frame = (
            str(case.first_frame)
            if case.first_frame is not None and case.first_frame.is_file()
            else None
        )
        all_frames, _num_frames = prepare(
            video_path=str(candidate_video),
            first_n_seconds=SAMPLE_DURATION,
            sample_fps=SAMPLE_FRAMES / SAMPLE_DURATION,
            max_frames=SAMPLE_FRAMES,
            start_seconds=0.0,
            gt_frame_path=gt_frame,
            label_prefix="frame",
            use_timestamp_labels=True,
        )

        judge = self._judge or judge_scene
        result = judge(
            rubric=rubric,
            all_frames=all_frames,
            provider=provider,
            model=model,
            alpha=alpha,
        )

        environment = getattr(result, "environment_result", None)
        objects = getattr(result, "object_result", None)
        metrics = {
            "overall_score": getattr(result, "overall_score", None),
            "env_score": getattr(environment, "score", None),
            "obj_score": getattr(objects, "score", None),
        }
        full_report = (
            result.model_dump() if hasattr(result, "model_dump") else dict(result)
        )
        return CaseResult(
            status=STATUS_COMPLETED,
            metrics=metrics,
            full_report=full_report,
        )
