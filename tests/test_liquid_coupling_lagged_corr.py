from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_lagged_correlation_recovers_lag():
    from evaluation.modules.liquid_coupling import compute_lagged_correlation

    fps = 30.0
    n = 200
    rng = np.random.default_rng(0)
    tilt = np.zeros(n, dtype=np.float32)
    tilt[50:150] = np.linspace(0, 90, 100, dtype=np.float32)
    pour_lag_frames = 6  # 0.2 s at 30 fps
    pour = np.zeros(n, dtype=np.float32)
    pour[50 + pour_lag_frames:150 + pour_lag_frames] = np.linspace(0, 200, 100, dtype=np.float32)
    pour += rng.normal(0, 1, size=pour.shape).astype(np.float32)
    out = compute_lagged_correlation(tilt=tilt, pour_rate=pour, fps=fps, max_lag_sec=1.0)
    assert out["lag_sec"] > 0
    assert abs(out["lag_sec"] - pour_lag_frames / fps) < 2.0 / fps
    assert out["correlation"] > 0.85


def test_lagged_correlation_random_returns_low():
    from evaluation.modules.liquid_coupling import compute_lagged_correlation

    rng = np.random.default_rng(1)
    n = 200
    tilt = rng.normal(0, 1, size=n).astype(np.float32)
    pour = rng.normal(0, 1, size=n).astype(np.float32)
    out = compute_lagged_correlation(tilt=tilt, pour_rate=pour, fps=30.0, max_lag_sec=1.0)
    assert abs(out["correlation"]) < 0.5
