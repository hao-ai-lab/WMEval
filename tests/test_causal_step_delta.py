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
