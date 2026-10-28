# WMEval/tests/test_slosh_spectrum.py
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _mask_with_centroid_y(H: int, W: int, cy: float, half_height: int = 8) -> np.ndarray:
    """Build a mask whose vertical centroid is at row `cy` (a thick horizontal band)."""
    m = np.zeros((H, W), dtype=bool)
    y0 = int(round(cy - half_height))
    y1 = int(round(cy + half_height))
    y0 = max(0, y0)
    y1 = min(H, y1)
    if y0 < y1:
        m[y0:y1, :] = True
    return m


def _oscillating_mask_sequence(T: int, freq_hz: float, fps: float, H: int = 80, W: int = 32) -> np.ndarray:
    cy_base = H / 2.0
    amp = 12.0
    t = np.arange(T) / fps
    cys = cy_base + amp * np.sin(2 * np.pi * freq_hz * t)
    return np.stack([_mask_with_centroid_y(H, W, cy) for cy in cys])


def test_dominant_frequency_matches_synthetic_oscillation():
    from evaluation.modules.slosh_spectrum import compute_slosh_spectrum_from_masks

    fps = 30.0
    T = 240  # 8 seconds — good resolution at low Hz
    freq_unity = 2.0  # Hz
    freq_wm = 2.0
    unity = _oscillating_mask_sequence(T, freq_unity, fps)
    wm = _oscillating_mask_sequence(T, freq_wm, fps)
    out = compute_slosh_spectrum_from_masks(unity_masks=unity, wm_masks=wm, fps=fps)

    # FFT bin width = fps / T = 0.125 Hz, so allow ±1 bin tolerance.
    assert out["dominant_frequency_hz"]["unity"] == pytest.approx(freq_unity, abs=0.13)
    assert out["dominant_frequency_hz"]["wm"] == pytest.approx(freq_wm, abs=0.13)
    assert out["dominant_frequency_hz"]["ratio"] == pytest.approx(1.0, abs=0.1)
    # Same waveforms ⇒ near-zero spectral L1 (centroid-extraction differences from
    # int rounding can create tiny residuals).
    assert out["spectral_l1"] < 0.05


def test_mismatched_frequencies_yield_ratio_far_from_one():
    from evaluation.modules.slosh_spectrum import compute_slosh_spectrum_from_masks

    fps = 30.0
    T = 240
    unity = _oscillating_mask_sequence(T, 2.0, fps)
    wm = _oscillating_mask_sequence(T, 4.0, fps)
    out = compute_slosh_spectrum_from_masks(unity_masks=unity, wm_masks=wm, fps=fps)

    assert out["dominant_frequency_hz"]["ratio"] == pytest.approx(2.0, rel=0.1)
    assert out["spectral_l1"] > 0.5  # very different shapes


def test_empty_masks_return_nan():
    from evaluation.modules.slosh_spectrum import compute_slosh_spectrum_from_masks

    T, H, W = 50, 32, 32
    empty = np.zeros((T, H, W), dtype=bool)
    out = compute_slosh_spectrum_from_masks(unity_masks=empty, wm_masks=empty, fps=30.0)
    assert np.isnan(out["dominant_frequency_hz"]["unity"])
    assert np.isnan(out["dominant_frequency_hz"]["wm"])
    assert np.isnan(out["dominant_frequency_hz"]["ratio"])


def test_too_short_sequence_returns_nan():
    from evaluation.modules.slosh_spectrum import compute_slosh_spectrum_from_masks

    masks = _oscillating_mask_sequence(T=3, freq_hz=2.0, fps=30.0)
    out = compute_slosh_spectrum_from_masks(unity_masks=masks, wm_masks=masks, fps=30.0)
    assert np.isnan(out["dominant_frequency_hz"]["unity"])


def _drifting_bottle_masks(T: int, fps: float, H: int = 80, W: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """A liquid trace with one clean slosh mode, plus a bottle-y trace that
    carries a *different* frequency — the bulk-motion signal the paper does
    not subtract from the slosh trace."""
    t = np.arange(T) / fps
    liquid = np.stack([_mask_with_centroid_y(H, W, H / 2.0 + 10.0 * np.sin(2 * np.pi * 2.0 * ti)) for ti in t])
    bottle_y = 3.0 * np.sin(2 * np.pi * 5.0 * t)
    return liquid, bottle_y


def test_slosh_trace_is_mean_centered_only_not_bulk_compensated():
    """B.4/C.4 mean-center the meniscus trace; bulk compensation is specified
    only for the velocity field. Passing a bottle trace must not move the
    canonical spectral_l1."""
    from evaluation.modules.slosh_spectrum import compute_slosh_spectrum_from_masks

    fps, T = 30.0, 240
    unity, bottle_u = _drifting_bottle_masks(T, fps)
    wm, bottle_w = _drifting_bottle_masks(T, fps)
    wm = _oscillating_mask_sequence(T, 3.0, fps)

    plain = compute_slosh_spectrum_from_masks(unity_masks=unity, wm_masks=wm, fps=fps)
    with_bottle = compute_slosh_spectrum_from_masks(
        unity_masks=unity, wm_masks=wm, fps=fps,
        unity_bottle_y=bottle_u, wm_bottle_y=bottle_w,
    )
    assert with_bottle["spectral_l1"] == pytest.approx(plain["spectral_l1"], abs=1e-12)
    assert with_bottle["meniscus"]["spectral_l1"] == pytest.approx(
        plain["meniscus"]["spectral_l1"], abs=1e-12
    )


def test_bulk_compensated_spectrum_is_kept_as_a_diagnostic():
    """The compensated figure is not the paper's metric, but it is what the
    suite reported before this fix, so it stays available under its own key."""
    from evaluation.modules.slosh_spectrum import compute_slosh_spectrum_from_masks

    fps, T = 30.0, 240
    unity, bottle_u = _drifting_bottle_masks(T, fps)
    wm = _oscillating_mask_sequence(T, 3.0, fps)
    _, bottle_w = _drifting_bottle_masks(T, fps)

    out = compute_slosh_spectrum_from_masks(
        unity_masks=unity, wm_masks=wm, fps=fps,
        unity_bottle_y=bottle_u, wm_bottle_y=bottle_w,
    )
    assert "bulk_compensated" in out
    assert out["bulk_compensated"]["spectral_l1"] != pytest.approx(
        out["spectral_l1"], abs=1e-9
    )
