# WMEval/tests/test_liquid_common_pour_onset.py
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_pour_onset_returns_none_on_flat_curve():
    from evaluation.modules._liquid_common import detect_pour_onset_from_curve

    curve = np.full(100, 0.7, dtype=np.float32)
    onset = detect_pour_onset_from_curve(curve, fps=30.0)
    assert onset is None


def test_pour_onset_finds_step_down():
    from evaluation.modules._liquid_common import detect_pour_onset_from_curve

    curve = np.full(100, 0.7, dtype=np.float32)
    # Frame 50 onwards: fill drops by 0.01 per frame for at least 5 frames.
    for i in range(50, 100):
        curve[i] = 0.7 - 0.01 * (i - 49)
    curve = np.clip(curve, 0.0, 1.0)
    onset = detect_pour_onset_from_curve(curve, fps=30.0, threshold_drop=0.005, k_consecutive=5)
    assert onset is not None
    # Onset is at frame 50, expressed as seconds: 50 / 30.
    assert abs(onset - 50.0 / 30.0) < 1.0 / 30.0


def test_pour_onset_ignores_brief_dip():
    from evaluation.modules._liquid_common import detect_pour_onset_from_curve

    curve = np.full(100, 0.7, dtype=np.float32)
    # 2-frame dip — should not register, k_consecutive=5
    curve[40] = 0.69
    curve[41] = 0.68
    onset = detect_pour_onset_from_curve(curve, fps=30.0, threshold_drop=0.005, k_consecutive=5)
    assert onset is None


def test_pour_onset_smoothing_robust_to_noise():
    """Noisy descending curve still finds onset near the true start."""
    from evaluation.modules._liquid_common import detect_pour_onset_from_curve

    rng = np.random.default_rng(0)
    curve = np.full(200, 0.7, dtype=np.float32)
    for i in range(80, 200):
        curve[i] = 0.7 - 0.005 * (i - 80)
    noise = rng.normal(0.0, 0.002, size=curve.shape).astype(np.float32)
    onset = detect_pour_onset_from_curve(
        curve + noise, fps=30.0, threshold_drop=0.003, k_consecutive=5, smoothing_window=5
    )
    assert onset is not None
    # Onset should be within ±10 frames of the true onset.
    assert abs(onset * 30.0 - 80) < 10
