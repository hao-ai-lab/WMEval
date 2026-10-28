# WMEval/tests/test_liquid_surface_orientation.py
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _flat_surface_mask(H: int, W: int, surface_row: int) -> np.ndarray:
    """Build a (H, W) mask that is True at and below `surface_row`."""
    m = np.zeros((H, W), dtype=bool)
    if 0 <= surface_row < H:
        m[surface_row:, :] = True
    return m


def _tilted_surface_mask(H: int, W: int, surface_row: int, slope: float) -> np.ndarray:
    """Mask where surface_y = surface_row + slope*(x - W/2). True at-or-below surface."""
    m = np.zeros((H, W), dtype=bool)
    cx = (W - 1) / 2.0
    for x in range(W):
        y_surface = int(round(surface_row + slope * (x - cx)))
        y_surface = max(0, min(y_surface, H - 1))
        m[y_surface:, x] = True
    return m


def test_flat_horizontal_surface_yields_zero_tilt():
    from evaluation.modules.liquid_surface_orientation import (
        compute_surface_orientation_from_masks,
    )

    H, W = 64, 64
    T = 10
    masks = np.stack([_flat_surface_mask(H, W, surface_row=20) for _ in range(T)])
    out = compute_surface_orientation_from_masks(unity_masks=masks, wm_masks=masks)

    assert out["surface_tilt_residual_deg"]["unity"] == pytest.approx(0.0, abs=1e-9)
    assert out["surface_tilt_residual_deg"]["wm"] == pytest.approx(0.0, abs=1e-9)
    assert out["surface_tilt_residual_deg"]["delta"] == pytest.approx(0.0, abs=1e-9)


def test_tilted_surface_recovers_expected_angle():
    from evaluation.modules.liquid_surface_orientation import (
        compute_surface_orientation_from_masks,
    )

    H, W = 128, 128
    T = 6
    slope = 0.25  # tan(angle) ⇒ angle ≈ 14.04 deg
    masks = np.stack([_tilted_surface_mask(H, W, surface_row=64, slope=slope) for _ in range(T)])
    flat = np.stack([_flat_surface_mask(H, W, surface_row=64) for _ in range(T)])
    out = compute_surface_orientation_from_masks(unity_masks=flat, wm_masks=masks)

    expected_deg = float(abs(np.degrees(np.arctan(slope))))
    assert out["surface_tilt_residual_deg"]["unity"] == pytest.approx(0.0, abs=1e-2)
    assert out["surface_tilt_residual_deg"]["wm"] == pytest.approx(expected_deg, rel=5e-2)
    assert out["surface_tilt_residual_deg"]["delta"] == pytest.approx(expected_deg, rel=5e-2)


def test_empty_mask_returns_nan():
    from evaluation.modules.liquid_surface_orientation import (
        compute_surface_orientation_from_masks,
    )

    H, W = 32, 32
    T = 5
    empty = np.zeros((T, H, W), dtype=bool)
    out = compute_surface_orientation_from_masks(unity_masks=empty, wm_masks=empty)
    assert np.isnan(out["surface_tilt_residual_deg"]["unity"])
    assert np.isnan(out["surface_tilt_residual_deg"]["wm"])
    assert np.isnan(out["surface_tilt_residual_deg"]["delta"])
