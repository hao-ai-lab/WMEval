# WMEval/tests/test_liquid_common_calibration.py
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _synthetic_mask_seq(fills: np.ndarray, h: int = 64, w: int = 32) -> np.ndarray:
    """Make T binary masks where the liquid is a rectangle of height fill*h."""
    masks = np.zeros((len(fills), h, w), dtype=bool)
    for i, f in enumerate(fills):
        rows = max(0, int(round(f * h)))
        if rows > 0:
            masks[i, h - rows:, :] = True
    return masks


def test_fit_fill_calibration_recovers_slope():
    from evaluation.modules._liquid_common import fit_fill_calibration

    fills = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    masks = _synthetic_mask_seq(fills)
    cal = fit_fill_calibration(unity_masks=masks, unity_log_fill=fills)
    # area = fill * 64 * 32 = fill * 2048; so slope ≈ 1/2048, intercept ≈ 0
    assert abs(cal.slope - 1.0 / 2048.0) < 1e-4
    assert abs(cal.intercept) < 1e-3
    assert cal.r2 > 0.99


def test_estimate_fill_curve_inverts_calibration():
    from evaluation.modules._liquid_common import (
        estimate_fill_curve_from_mask,
        fit_fill_calibration,
    )

    fills = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    masks = _synthetic_mask_seq(fills)
    cal = fit_fill_calibration(unity_masks=masks, unity_log_fill=fills)

    new_fills = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    new_masks = _synthetic_mask_seq(new_fills)
    estimated = estimate_fill_curve_from_mask(new_masks, calibration=cal)
    assert estimated.shape == new_fills.shape
    assert np.allclose(estimated, new_fills, atol=0.02)
