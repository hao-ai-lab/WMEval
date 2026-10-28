# WMEval/tests/test_liquid_common_divcurl.py
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_field(u: np.ndarray, v: np.ndarray) -> "PIVField":
    from evaluation.modules._liquid_common import PIVField

    return PIVField(u=u.astype(np.float32), v=v.astype(np.float32), valid=np.ones_like(u, dtype=bool))


def test_divergence_zero_for_uniform_field():
    from evaluation.modules._liquid_common import divergence_curl_2d

    h, w = 16, 16
    u = np.ones((h, w), dtype=np.float32)
    v = np.full((h, w), 2.0, dtype=np.float32)
    div, curl = divergence_curl_2d(_make_field(u, v))
    inner = (slice(1, -1), slice(1, -1))
    assert np.allclose(div[inner], 0.0, atol=1e-5)
    assert np.allclose(curl[inner], 0.0, atol=1e-5)


def test_curl_for_pure_rotation():
    """For v(x,y) = (-y, x), curl ≡ ∂v/∂x − ∂u/∂y = 1 − (−1) = 2."""
    from evaluation.modules._liquid_common import divergence_curl_2d

    h, w = 32, 32
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = -yy
    v = xx
    div, curl = divergence_curl_2d(_make_field(u, v))
    inner = (slice(2, -2), slice(2, -2))
    assert np.allclose(div[inner], 0.0, atol=1e-3)
    assert np.allclose(curl[inner], 2.0, atol=1e-3)


def test_divergence_for_pure_sink():
    """For v(x,y) = (-x, -y), div ≡ ∂u/∂x + ∂v/∂y = -1 + -1 = -2."""
    from evaluation.modules._liquid_common import divergence_curl_2d

    h, w = 32, 32
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = -xx
    v = -yy
    div, _curl = divergence_curl_2d(_make_field(u, v))
    inner = (slice(2, -2), slice(2, -2))
    assert np.allclose(div[inner], -2.0, atol=1e-3)


def test_invalid_pixels_propagate_as_nan():
    """A single invalid pixel at (8,8) must NaN-mask itself and its 4 cardinal
    stencil neighbours, but NOT pixels two steps away."""
    from evaluation.modules._liquid_common import PIVField, divergence_curl_2d

    h, w = 16, 16
    u = np.ones((h, w), dtype=np.float32)
    v = np.ones((h, w), dtype=np.float32)
    valid = np.ones_like(u, dtype=bool)
    valid[8, 8] = False
    field = PIVField(u=u, v=v, valid=valid)
    div, curl = divergence_curl_2d(field)
    # Self
    assert np.isnan(div[8, 8])
    assert np.isnan(curl[8, 8])
    # Cardinal neighbours — must also be NaN because the central-difference
    # stencil at each of these reads pixel (8, 8).
    for r, c in [(7, 8), (9, 8), (8, 7), (8, 9)]:
        assert np.isnan(div[r, c]), f"div[{r},{c}] should be NaN (stencil reads (8,8))"
        assert np.isnan(curl[r, c]), f"curl[{r},{c}] should be NaN"
    # Two steps away — must NOT be NaN; the stencil there does not touch (8, 8).
    for r, c in [(8, 6), (8, 10), (6, 8), (10, 8)]:
        assert not np.isnan(div[r, c]), f"div[{r},{c}] should be finite (stencil does not touch (8,8))"
        assert not np.isnan(curl[r, c]), f"curl[{r},{c}] should be finite"
