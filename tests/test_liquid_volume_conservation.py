# WMEval/tests/test_liquid_volume_conservation.py
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _square_mask_sequence(area_per_frame: np.ndarray, H: int = 64, W: int = 64) -> np.ndarray:
    """Build (T, H, W) bool masks where mask[t] is a square with `area_per_frame[t]` pixels.

    The square is centered. For non-perfect squares, fill row-by-row to hit the area exactly.
    """
    T = int(area_per_frame.shape[0])
    masks = np.zeros((T, H, W), dtype=bool)
    for t in range(T):
        a = int(round(float(area_per_frame[t])))
        a = max(0, min(a, H * W))
        if a == 0:
            continue
        # Fill the first `a` cells in raster order — area is exactly a.
        flat = masks[t].reshape(-1)
        flat[:a] = True
        masks[t] = flat.reshape(H, W)
    return masks


def test_perfectly_stable_masks_have_zero_std_zero_drift():
    from evaluation.modules.liquid_volume_conservation import (
        compute_volume_conservation_from_masks,
    )

    T = 30
    fps = 30.0
    area = np.full(T, 1000.0, dtype=np.float64)
    unity = _square_mask_sequence(area)
    wm = _square_mask_sequence(area)
    out = compute_volume_conservation_from_masks(unity_masks=unity, wm_masks=wm, fps=fps)

    assert out["mask_area_std_normalized"]["unity"] == pytest.approx(0.0, abs=1e-9)
    assert out["mask_area_std_normalized"]["wm"] == pytest.approx(0.0, abs=1e-9)
    assert out["mask_area_std_normalized"]["delta"] == pytest.approx(0.0, abs=1e-9)
    assert out["mask_area_drift_per_sec"]["unity"] == pytest.approx(0.0, abs=1e-9)
    assert out["mask_area_drift_per_sec"]["wm"] == pytest.approx(0.0, abs=1e-9)
    assert out["mask_area_drift_per_sec"]["delta"] == pytest.approx(0.0, abs=1e-9)


def test_linear_growth_yields_positive_drift_matches_slope_over_mean():
    from evaluation.modules.liquid_volume_conservation import (
        compute_volume_conservation_from_masks,
    )

    T = 30
    fps = 30.0  # so total duration = 1.0 sec
    # area grows from 1000 → 1290 pixels: slope = 10 px/frame = 300 px/sec, mean = 1145
    area = 1000.0 + 10.0 * np.arange(T, dtype=np.float64)
    unity = _square_mask_sequence(np.full(T, 1000.0))  # stable
    wm = _square_mask_sequence(area)
    out = compute_volume_conservation_from_masks(unity_masks=unity, wm_masks=wm, fps=fps)

    expected_slope_per_sec = 10.0 * fps  # 300 px/sec
    expected_mean = float(area.mean())
    expected_drift = expected_slope_per_sec / expected_mean

    assert out["mask_area_drift_per_sec"]["unity"] == pytest.approx(0.0, abs=1e-9)
    assert out["mask_area_drift_per_sec"]["wm"] == pytest.approx(expected_drift, rel=1e-6)
    assert out["mask_area_drift_per_sec"]["delta"] == pytest.approx(expected_drift, rel=1e-6)


def test_oscillating_area_has_nonzero_std_zero_drift():
    from evaluation.modules.liquid_volume_conservation import (
        compute_volume_conservation_from_masks,
    )

    T = 60
    fps = 30.0
    t = np.arange(T, dtype=np.float64) / fps
    area = 1000.0 + 50.0 * np.sin(2 * np.pi * 2.0 * t)
    unity = _square_mask_sequence(area)
    wm = _square_mask_sequence(area)
    out = compute_volume_conservation_from_masks(unity_masks=unity, wm_masks=wm, fps=fps)

    expected_std_norm = float(area.std() / area.mean())
    assert out["mask_area_std_normalized"]["unity"] == pytest.approx(expected_std_norm, rel=1e-3)
    assert out["mask_area_std_normalized"]["delta"] == pytest.approx(0.0, abs=1e-6)
    # Pure oscillation around constant mean ⇒ near-zero linear drift.
    # Tolerance accommodates the integer-rounding bias in `_square_mask_sequence`.
    assert abs(out["mask_area_drift_per_sec"]["unity"]) < 0.05


def test_zero_mean_mask_returns_nan_without_crashing():
    from evaluation.modules.liquid_volume_conservation import (
        compute_volume_conservation_from_masks,
    )

    T = 5
    H = W = 8
    masks = np.zeros((T, H, W), dtype=bool)
    out = compute_volume_conservation_from_masks(unity_masks=masks, wm_masks=masks, fps=10.0)
    assert np.isnan(out["mask_area_std_normalized"]["unity"])
    assert np.isnan(out["mask_area_drift_per_sec"]["unity"])
