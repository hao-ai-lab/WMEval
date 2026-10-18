"""Adapter that drives the prompt-consistency VLM judge.

Wraps the existing ``evaluation.modules.vlm_consistency`` pipeline (PR #14 /
PR #17) as a suite dimension WITHOUT touching its internals: rubrics are read
from the case bundle's committed rubric cache (``vlm.rubric_dir`` is
case-relative, default ``reference/artifacts/rubrics`` — inside the
checksummed bundle, so the judging basis is tamper-evident) via
``find_rubric`` (missing rubric → ``missing_rubric`` row, never generated
inline — generation stays an offline authoring step), frames are sampled with
the exact ``batch.py`` constants, and scoring goes through ``judge_scene``.

The Unity reference video is judged once, offline
(``tools/freeze_prompt_reference_scores.py``), and its scores ship in the
bundle as ``reference/artifacts/reference_scores.json``. Each completed row
reports them as ``<score>_unity`` beside the candidate's own scores, but only
while the judge settings recorded in that file still match the live ones;
otherwise the row says the reference is stale and leaves them out.

The heavyweight imports (cv2, pydantic, provider clients) happen inside
``evaluate`` so dry runs and suite loading stay dependency-light.
"""
from __future__ import annotations

import json
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

# reference.artifacts key and file holding the frozen Unity reference scores.
REFERENCE_SCORES_ARTIFACT = "reference_scores"
REFERENCE_SCORES_FILENAME = "reference_scores.json"
REFERENCE_SCORES_SCHEMA_VERSION = 1
SCORE_KEYS = ("overall_score", "env_score", "obj_score")


def rubric_scene_name(case: LiteCase) -> str:
    """The scene name rubrics are cached under (PR #17 kept the original Unity
    scene folder names, e.g. ``MirrorRollingBall_first``)."""
    return case.rubric_scene_name or case.scene


def judge_identity(
    dimension: DimensionSpec, case: LiteCase, rubric_path: Path
) -> dict[str, Any]:
    """Everything that decides a score besides the video itself. Two scores
    are comparable only when these match."""
    from evaluation.lite.authoring import sha256_file
    from evaluation.modules.vlm_consistency.batch import (
        SAMPLE_DURATION,
        SAMPLE_FRAMES,
    )
    from evaluation.modules.vlm_consistency.vlm_judge import SYSTEM_PROMPT_SHA256

    cfg = dimension.vlm
    first_frame = (
        case.first_frame
        if case.first_frame is not None and case.first_frame.is_file()
        else None
    )
    return {
        "provider": str(cfg["provider"]),
        "model": str(cfg["model"]),
        "prompt_version": str(cfg["prompt_version"]),
        "alpha": float(cfg["alpha"]),
        "rubric": rubric_path.resolve().relative_to(case.root).as_posix(),
        "rubric_sha256": sha256_file(rubric_path),
        "first_frame_sha256": sha256_file(first_frame) if first_frame else None,
        "sample_duration_sec": SAMPLE_DURATION,
        "sample_frames": SAMPLE_FRAMES,
        "system_prompt_sha256": dict(SYSTEM_PROMPT_SHA256),
    }


def load_reference_scores(
    case: LiteCase, identity: dict[str, Any]
) -> dict[str, Any]:
    """The case's frozen Unity reference scores, checked against ``identity``.

    Returns ``status`` ``ok`` (with ``scores``), ``not_frozen`` (the case
    declares no file) or ``stale`` (with the ``mismatches`` that make the
    frozen scores incomparable to a live score).
    """
    path = case.reference_artifacts.get(REFERENCE_SCORES_ARTIFACT)
    if path is None:
        return {"status": "not_frozen"}
    from evaluation.lite.authoring import sha256_file

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != REFERENCE_SCORES_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported schema_version {document.get('schema_version')!r}"
        )
    if document.get("case_id") != case.case_id:
        raise ValueError(f"{path}: written for case {document.get('case_id')!r}")

    frozen = document.get("judge") or {}
    mismatches = sorted(
        key
        for key in set(identity) | set(frozen)
        if identity.get(key) != frozen.get(key)
    )
    video = case.root / str(document.get("reference_video", ""))
    if not video.is_file() or sha256_file(video) != document.get("reference_video_sha256"):
        mismatches.append("reference_video_sha256")
    summary = {
        "file": Path(path).relative_to(case.root).as_posix(),
        "num_passes": len(document.get("passes") or []),
        "judge_model_resolved": document.get("judge_model_resolved"),
    }
    if mismatches:
        return {"status": "stale", "mismatches": mismatches, **summary}
    scores = {
        key: (document.get("summary") or {}).get(key, {}).get("mean")
        for key in SCORE_KEYS
    }
    return {"status": "ok", "scores": scores, **summary}


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
        result = self.judge_video(dimension=dimension, case=case, video=candidate_video)
        if result.status != STATUS_COMPLETED:
            return result
        reference = load_reference_scores(case, result.full_report["judge"])
        result.full_report["reference_scores"] = reference
        if reference["status"] == "ok":
            for key, value in reference["scores"].items():
                result.metrics[f"{key}_unity"] = value
        return result

    def judge_video(
        self, *, dimension: DimensionSpec, case: LiteCase, video: Path
    ) -> CaseResult:
        """One judge pass over ``video`` with the dimension's settings. Used
        for candidates and, offline, for the case's Unity reference."""
        cfg = dimension.vlm
        provider = str(cfg["provider"])
        model = str(cfg["model"])
        prompt_version = str(cfg["prompt_version"])
        alpha = float(cfg["alpha"])
        # case-relative: the rubric ships inside the case's checksummed bundle
        rubric_dir = case.root / str(cfg["rubric_dir"])

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
            video_path=str(video),
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
        full_report["judge"] = judge_identity(dimension, case, Path(rubric_path))
        return CaseResult(
            status=STATUS_COMPLETED,
            metrics=metrics,
            full_report=full_report,
        )
