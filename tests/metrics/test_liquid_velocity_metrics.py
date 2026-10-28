# WMEval/tests/test_liquid_velocity_metrics.py
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_field(u, v, valid=None):
    from evaluation.modules._liquid_common import PIVField

    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if valid is None:
        valid = np.ones_like(u, dtype=bool)
    return PIVField(u=u, v=v, valid=valid)


def test_velocity_metrics_identity():
    from evaluation.modules.liquid_velocity import compute_velocity_metrics_from_fields

    rng = np.random.default_rng(0)
    u = rng.normal(0, 1, size=(5, 8, 8)).astype(np.float32)
    v = rng.normal(0, 1, size=(5, 8, 8)).astype(np.float32)
    valid = np.ones_like(u, dtype=bool)
    f_u = _make_field(u, v, valid)
    f_w = _make_field(u, v, valid)
    out = compute_velocity_metrics_from_fields(unity=f_u, wm=f_w)
    assert out["epe"]["mean"] < 1e-5
    assert out["cosine_similarity"] > 0.999
    assert out["temporal_consistency"] == 1.0


def test_velocity_metrics_constant_offset():
    """If wm = unity + (1, 0), EPE mean = 1.0, cosine similarity < 1 (eps biases parallel vectors below 1)."""
    from evaluation.modules.liquid_velocity import compute_velocity_metrics_from_fields

    u = np.ones((4, 6, 6), dtype=np.float32)
    v = np.zeros((4, 6, 6), dtype=np.float32)
    f_u = _make_field(u, v)
    f_w = _make_field(u + 1.0, v)
    out = compute_velocity_metrics_from_fields(unity=f_u, wm=f_w)
    assert abs(out["epe"]["mean"] - 1.0) < 1e-5
    assert out["cosine_similarity"] < 1.0


def test_cosine_similarity_weights_every_frame_equally():
    """Eq. 14 is a nested expectation E_t E_{p in V_t}, not a flat mean.

    Frame 0 has 4 valid cells all perfectly aligned (cos = 1); frame 1 has a
    single valid cell that is anti-aligned (cos = -1). Averaging over frames
    gives (1 + -1)/2 = 0. Flattening over all five valid cells instead gives
    (4*1 + -1)/5 = +0.6, which over-weights whichever frame happens to expose
    more liquid — exactly what varies most under chirp_strafe.
    """
    from evaluation.modules.liquid_velocity import compute_velocity_metrics_from_fields

    u_unity = np.array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]])
    v_unity = np.zeros_like(u_unity)
    u_wm = np.array([[[1.0, 1.0], [1.0, 1.0]], [[-1.0, 0.0], [0.0, 0.0]]])
    v_wm = np.zeros_like(u_wm)
    valid = np.array(
        [[[True, True], [True, True]], [[True, False], [False, False]]]
    )

    out = compute_velocity_metrics_from_fields(
        unity=_make_field(u_unity, v_unity, valid),
        wm=_make_field(u_wm, v_wm, valid),
    )

    assert abs(out["cosine_similarity"]) < 1e-6


def test_cosine_similarity_skips_frames_with_no_valid_cells():
    """A frame where the liquid mask is empty contributes no term at all,
    rather than dragging the per-frame mean toward zero."""
    from evaluation.modules.liquid_velocity import compute_velocity_metrics_from_fields

    u = np.array([[[1.0, 1.0]], [[1.0, 1.0]]])
    v = np.zeros_like(u)
    valid = np.array([[[True, True]], [[False, False]]])

    out = compute_velocity_metrics_from_fields(
        unity=_make_field(u, v, valid),
        wm=_make_field(u, v, valid),
    )

    assert out["cosine_similarity"] > 0.999
