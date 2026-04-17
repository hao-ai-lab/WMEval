# tests/test_causal_step_delta.py
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAUSALSTEP_ROOT = ROOT.parent / "CausalStep"
if str(CAUSALSTEP_ROOT) not in sys.path:
    sys.path.insert(0, str(CAUSALSTEP_ROOT))


def test_causalstep_package_importable() -> None:
    from causalstep.schema import VideoInstance, QAPair, Segment, AggregateResults
    from causalstep.model_interface import PrecomputedModel
    from causalstep.evaluator import evaluate_chain, evaluate_descriptive_isolated, evaluate_causal_isolated
    from causalstep.metrics import compute_all_metrics

    assert VideoInstance is not None
    assert PrecomputedModel is not None
    assert evaluate_chain is not None


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "beamdomino"


def _load_tiny_videos():
    from causalstep.schema import load_dataset
    return load_dataset(str(FIXTURE_DIR / "tiny_qa_dataset.json"))


def _evaluate_with_answers(video, answer_map):
    from causalstep.model_interface import PrecomputedModel
    from causalstep.evaluator import (
        evaluate_chain, evaluate_descriptive_isolated, evaluate_causal_isolated,
    )
    from causalstep.metrics import compute_all_metrics

    keyed = {(video.video_id, idx, qt): ans for (idx, qt), ans in answer_map.items()}
    model = PrecomputedModel(keyed, video_id=video.video_id)
    chain = evaluate_chain(video, model)
    desc = evaluate_descriptive_isolated(video, model)
    causal = evaluate_causal_isolated(video, model)
    return compute_all_metrics([chain], [desc], [causal])


def _perfect_answers():
    return {
        (1, "descriptive"): "A", (2, "descriptive"): "B",
        (3, "descriptive"): "C", (4, "descriptive"): "D",
        (2, "causal"): "A", (3, "causal"): "A", (4, "causal"): "A",
    }


def test_perfect_chain_metrics() -> None:
    (video,) = _load_tiny_videos()
    results = _evaluate_with_answers(video, _perfect_answers())
    assert results.csr == 100.0
    assert results.amcl == 4.0
    assert results.mcl == 4
    assert results.rf == 0.0
    assert results.dua == 100.0
    assert results.icra == 100.0
    # WS weighted score: desc=1 then causal scores 2, 3, 4 -> 1+2+3+4 = 10
    assert results.ws == 10.0


def test_all_wrong_metrics() -> None:
    (video,) = _load_tiny_videos()
    wrong = {k: "Z" for k in _perfect_answers()}
    results = _evaluate_with_answers(video, wrong)
    assert results.csr == 0.0
    assert results.mcl == 0
    assert results.amcl == 0.0
    assert results.dua == 0.0
    assert results.icra == 0.0
    assert results.ws == 0.0


def test_chain_breaks_at_causal_3() -> None:
    (video,) = _load_tiny_videos()
    answers = _perfect_answers()
    answers[(3, "causal")] = "B"  # wrong
    results = _evaluate_with_answers(video, answers)
    assert results.mcl >= 2
    assert results.dua == 100.0
    assert round(results.icra, 1) == 66.7
    assert results.rf >= 1.0


def test_compute_delta_higher_is_better_metrics():
    from evaluation.modules._causal_step_delta import compute_delta

    unity = {"csr": 100.0, "amcl": 4.0, "mcl": 4, "rf": 0.0, "ws": 10.0, "dua": 100.0, "icra": 100.0}
    wm    = {"csr":   0.0, "amcl": 2.0, "mcl": 2, "rf": 1.0, "ws":  3.0, "dua":  75.0, "icra":  66.7}
    delta = compute_delta(unity, wm)
    assert delta["csr"] == 100.0
    assert delta["amcl"] == 2.0
    assert delta["mcl"] == 2
    assert delta["ws"] == 7.0
    assert delta["dua"] == 25.0
    assert round(delta["icra"], 1) == 33.3


def test_compute_delta_rf_is_inverted():
    from evaluation.modules._causal_step_delta import compute_delta

    unity = {"csr": 100.0, "amcl": 4.0, "mcl": 4, "rf": 0.0, "ws": 10.0, "dua": 100.0, "icra": 100.0}
    wm    = {"csr":   0.0, "amcl": 2.0, "mcl": 2, "rf": 1.0, "ws":  3.0, "dua":  75.0, "icra":  66.7}
    delta = compute_delta(unity, wm)
    assert delta["rf"] == 1.0


def test_compute_delta_convention_string():
    from evaluation.modules._causal_step_delta import DELTA_CONVENTION_STRING
    assert "positive = wm worse" in DELTA_CONVENTION_STRING.lower()
