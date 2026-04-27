# WMEval/tests/test_liquid_recorder_schema.py
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "tests" / "fixtures" / "liquid" / "unity_gt_log_tiny.json"


def test_load_unity_log_tiny():
    from evaluation.modules._liquid_common import load_unity_log

    log = load_unity_log(str(FIXTURE))
    assert log.fps == 30.0
    assert log.scene == "liquid_container"
    assert log.bottle_local_volume_ml == 500.0
    assert log.num_frames == 2
    assert np.allclose(log.fill_fraction, [0.7, 0.69], atol=1e-3)
    assert np.allclose(log.tilt_deg, [0.0, 60.0], atol=1e-3)
    assert log.is_pouring.tolist() == [False, True]


def test_load_unity_log_missing_required_key_raises(tmp_path):
    from evaluation.modules._liquid_common import load_unity_log

    bad = json.loads(FIXTURE.read_text())
    del bad["fps"]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="missing required key: fps"):
        load_unity_log(str(p))


def test_load_unity_log_missing_frame_field_raises(tmp_path):
    from evaluation.modules._liquid_common import load_unity_log

    bad = json.loads(FIXTURE.read_text())
    del bad["frames"][0]["liquid"]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="frame 0 missing 'liquid'"):
        load_unity_log(str(p))
