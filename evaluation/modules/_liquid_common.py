# WMEval/evaluation/modules/_liquid_common.py
"""Shared helpers for the LiquidContainer eval modules.

This module provides:
  - Schema dataclasses (LiquidMaskResult, PIVField, FillCalibration)
  - Pure-numerical helpers (divergence_curl_2d, fit_fill_calibration,
    estimate_fill_curve_from_mask, detect_pour_onset_from_curve)
  - Cached I/O wrappers around SAM2 (get_liquid_mask_video) and
    OpenPIV (get_piv_field)

The cache key is (video_path, content_hash, config_hash) where config_hash
covers only the keys that affect each output. Changing PIV settings does
not invalidate SAM2 caches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class LiquidMaskResult:
    """Per-frame binary masks for the liquid region inside the bottle.

    masks[t, h, w] is True where the liquid (NOT including the glass / bottle)
    occupies the pixel.
    """

    masks: np.ndarray  # (T, H, W) bool
    fps: float
    sam2_model: str
    prompt: str

    @property
    def num_frames(self) -> int:
        return int(self.masks.shape[0])


@dataclass
class PIVField:
    """2D velocity field (u right-positive, v down-positive in image coords).

    For an OpenPIV computation with window size W and overlap O, the spatial
    grid is downsampled to (T, H//(W-O), W//(W-O)). For convenience we keep
    these as the field's H, W (they will be smaller than image H, W).
    """

    u: np.ndarray  # (..., H, W) float32, px/frame
    v: np.ndarray  # (..., H, W) float32, px/frame
    valid: np.ndarray  # (..., H, W) bool — False where PIV failed


@dataclass
class FillCalibration:
    """Linear fit mask_area_px → fill_fraction in [0, 1]."""

    slope: float
    intercept: float
    r2: float
    method: str = "linear_robust"


def divergence_curl_2d(field: PIVField) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference 2D divergence and curl on the (H, W) plane.

    Operates on the last two axes if u, v have a leading time axis. Invalid
    pixels (and any neighbour participating in the central-difference stencil
    that is invalid) propagate as NaN. The borders are also NaN.
    """
    u = field.u.astype(np.float32, copy=False)
    v = field.v.astype(np.float32, copy=False)
    valid = field.valid

    if u.shape != v.shape or u.shape != valid.shape:
        raise ValueError(
            f"divergence_curl_2d: shape mismatch u={u.shape} v={v.shape} valid={valid.shape}"
        )

    # Central differences on last two axes.
    du_dx = np.full_like(u, np.nan)
    du_dy = np.full_like(u, np.nan)
    dv_dx = np.full_like(v, np.nan)
    dv_dy = np.full_like(v, np.nan)

    du_dx[..., :, 1:-1] = 0.5 * (u[..., :, 2:] - u[..., :, :-2])
    du_dy[..., 1:-1, :] = 0.5 * (u[..., 2:, :] - u[..., :-2, :])
    dv_dx[..., :, 1:-1] = 0.5 * (v[..., :, 2:] - v[..., :, :-2])
    dv_dy[..., 1:-1, :] = 0.5 * (v[..., 2:, :] - v[..., :-2, :])

    div = du_dx + dv_dy
    curl = dv_dx - du_dy

    # Mask: a pixel is valid if itself AND its left/right/up/down neighbours are valid.
    stencil_ok = np.zeros_like(valid)
    stencil_ok[..., 1:-1, 1:-1] = (
        valid[..., 1:-1, 1:-1]
        & valid[..., 1:-1, :-2]
        & valid[..., 1:-1, 2:]
        & valid[..., :-2, 1:-1]
        & valid[..., 2:, 1:-1]
    )
    div = np.where(stencil_ok, div, np.nan)
    curl = np.where(stencil_ok, curl, np.nan)
    return div, curl
