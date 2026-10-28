from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.lite.adapters.registry_adapter import (
    ARTIFACT_REFERENCE_ANSWERS,
    build_runner_kwargs,
    merge_metric_config,
)
from evaluation.lite.candidate_discovery import CandidateRow
from evaluation.lite.case_loader import LiteCase
from evaluation.lite.suite_loader import DimensionSpec


def _dimension(**overrides) -> DimensionSpec:
    kwargs = dict(
        dimension_id="physical_causality",
        display_name="Physical Causality",
        description="",
        evaluator="registry",
        requires_reference=True,
        cases=("BeamDomino/causal",),
        modules=("causal_step",),
        metric_config={},
    )
    kwargs.update(overrides)
    return DimensionSpec(**kwargs)


def _case(tmp_path: Path, *, artifacts: dict) -> LiteCase:
    return LiteCase(
        root=tmp_path,
        schema_version=2,
        case_id="BeamDomino/causal",
        scene="BeamDomino",
        display_name="BeamDomino",
        modules=("causal_step",),
        reference_video=tmp_path / "reference.mp4",
        baseline_path=None,
        checksums_path=tmp_path / "checksums.sha256",
        protocol={},
        evaluator_lock={},
        reference_artifacts={k: Path(v) for k, v in artifacts.items()},
    )


def _candidate(tmp_path: Path) -> CandidateRow:
    return CandidateRow(
        model="m",
        case_id="BeamDomino/causal",
        seed="0",
        candidate_video=tmp_path / "cand.mp4",
    )


def test_judge_repeats_is_an_accepted_metric_config_key(tmp_path: Path):
    """Repeat count has to be settable from the dimension block, otherwise the
    suite cannot ask for the paper's five passes."""
    dim = _dimension(metric_config={"causal_step_judge_repeats": 5})
    merged = merge_metric_config(
        dim, _case(tmp_path, artifacts={"qa_dataset": tmp_path / "qa.json"}), _candidate(tmp_path)
    )
    assert merged["causal_step_judge_repeats"] == 5


def test_reference_answers_artifact_is_passed_to_the_runner(tmp_path: Path):
    rec = tmp_path / "judge_answers.json"
    kwargs = build_runner_kwargs(
        dimension=_dimension(),
        case=_case(
            tmp_path,
            artifacts={"qa_dataset": tmp_path / "qa.json", ARTIFACT_REFERENCE_ANSWERS: rec},
        ),
        candidate=_candidate(tmp_path),
        candidate_video=tmp_path / "cand.mp4",
        device=None,
    )
    assert kwargs["causal_step_reference_answers"] == str(rec)


def test_reference_answers_artifact_is_optional(tmp_path: Path):
    """A case with no recording still runs — it just pays the live judge."""
    kwargs = build_runner_kwargs(
        dimension=_dimension(),
        case=_case(tmp_path, artifacts={"qa_dataset": tmp_path / "qa.json"}),
        candidate=_candidate(tmp_path),
        candidate_video=tmp_path / "cand.mp4",
        device=None,
    )
    assert "causal_step_reference_answers" not in kwargs


def test_runner_signature_accepts_the_new_causal_kwargs():
    """Whitelisting a key is only half the wiring — `run_world_model_evaluation`
    must also accept it, or the lite path raises TypeError at call time."""
    import inspect

    from evaluation.runner import run_world_model_evaluation

    params = inspect.signature(run_world_model_evaluation).parameters
    assert "causal_step_judge_repeats" in params
    assert "causal_step_reference_answers" in params


def test_runner_assigns_the_new_causal_kwargs_into_context_artifacts():
    """context.artifacts is the only channel causal_step reads config from, so
    a kwarg that never reaches it is inert even though the signature accepts it."""
    import inspect

    from evaluation.runner import run_world_model_evaluation

    src = inspect.getsource(run_world_model_evaluation)
    assert 'context.artifacts["causal_step_judge_repeats"]' in src
    assert 'context.artifacts["causal_step_reference_answers"]' in src
