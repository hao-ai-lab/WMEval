# WMEval/tests/test_liquid_state_metrics.py
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_curves(unity_fill, wm_fill, fps=30.0):
    return np.asarray(unity_fill, dtype=np.float32), np.asarray(wm_fill, dtype=np.float32), fps


def test_fill_curve_correlation_identity_perfect():
    from evaluation.modules.liquid_state import compute_state_metrics_from_arrays

    unity_fill, wm_fill, fps = _make_curves(np.linspace(0.7, 0.2, 60), np.linspace(0.7, 0.2, 60))
    out = compute_state_metrics_from_arrays(
        unity_fill=unity_fill, wm_fill=wm_fill, fps=fps,
        unity_pour_rate=np.zeros(60, dtype=np.float32),
        wm_pour_rate=np.zeros(60, dtype=np.float32),
        unity_surface_height=unity_fill,
        wm_surface_height=wm_fill,
        unity_spilled_final_ml=150.0, wm_spilled_estimate_ml=150.0,
    )
    assert out["fill"]["curve_correlation"] > 0.999
    assert out["fill"]["mae"] < 1e-5
    assert abs(out["spill"]["final_volume_error_ml"]) < 1e-5
    assert out["surface"]["height_trace_correlation"] > 0.999


def test_pour_onset_offset_signed():
    from evaluation.modules.liquid_state import compute_state_metrics_from_arrays

    fps = 30.0
    n = 90
    unity_fill = np.full(n, 0.7, dtype=np.float32)
    wm_fill = np.full(n, 0.7, dtype=np.float32)
    # Unity pours starting frame 30 (1 s), WM pours starting frame 36 (1.2 s, 6 frames late)
    for i in range(30, n):
        unity_fill[i] = 0.7 - 0.01 * (i - 30)
    for i in range(36, n):
        wm_fill[i] = 0.7 - 0.01 * (i - 36)
    out = compute_state_metrics_from_arrays(
        unity_fill=unity_fill, wm_fill=wm_fill, fps=fps,
        unity_pour_rate=np.zeros(n, dtype=np.float32),
        wm_pour_rate=np.zeros(n, dtype=np.float32),
        unity_surface_height=unity_fill, wm_surface_height=wm_fill,
        unity_spilled_final_ml=200.0, wm_spilled_estimate_ml=200.0,
    )
    expected = (36 - 30) / fps  # WM is 6 frames late → +0.2 s
    assert abs(out["pour"]["onset_offset_sec"] - expected) < 1.5 / fps


def test_pour_rate_dtw_zero_for_identity():
    from evaluation.modules.liquid_state import compute_state_metrics_from_arrays

    fps = 30.0
    rate = np.concatenate([np.zeros(30), np.linspace(0, 200, 30), np.linspace(200, 0, 30)]).astype(np.float32)
    out = compute_state_metrics_from_arrays(
        unity_fill=np.linspace(0.7, 0.4, 90, dtype=np.float32),
        wm_fill=np.linspace(0.7, 0.4, 90, dtype=np.float32),
        fps=fps,
        unity_pour_rate=rate, wm_pour_rate=rate,
        unity_surface_height=np.zeros(90, dtype=np.float32),
        wm_surface_height=np.zeros(90, dtype=np.float32),
        unity_spilled_final_ml=100.0, wm_spilled_estimate_ml=100.0,
    )
    assert out["pour"]["rate_curve_dtw"] < 1e-5


def test_pour_rate_dtw_nonzero_for_shifted_rate():
    from evaluation.modules.liquid_state import compute_state_metrics_from_arrays

    fps = 30.0
    rate_unity = np.concatenate([np.zeros(30), np.linspace(0, 200, 30), np.linspace(200, 0, 30)]).astype(np.float32)
    rate_wm = np.zeros_like(rate_unity)
    rate_wm[30:60] = 50.0  # very different shape
    rate_wm[60:90] = 50.0
    out = compute_state_metrics_from_arrays(
        unity_fill=np.linspace(0.7, 0.4, 90, dtype=np.float32),
        wm_fill=np.linspace(0.7, 0.55, 90, dtype=np.float32),
        fps=fps,
        unity_pour_rate=rate_unity, wm_pour_rate=rate_wm,
        unity_surface_height=np.zeros(90, dtype=np.float32),
        wm_surface_height=np.zeros(90, dtype=np.float32),
        unity_spilled_final_ml=100.0, wm_spilled_estimate_ml=20.0,
    )
    assert out["pour"]["rate_curve_dtw"] > 0.1
    assert out["spill"]["final_volume_error_ml"] == pytest.approx(80.0)
