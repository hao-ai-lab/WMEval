# WMEval/tests/test_liquid_velocity_metrics.py
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
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
