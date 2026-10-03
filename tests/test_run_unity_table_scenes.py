from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import tools.run_unity_table_scenes as module


def _racing_capture(path: Path, *, mass: float | None) -> None:
    (path / "frames").mkdir(parents=True)
    (path / "frames/frame_0000.png").write_bytes(b"placeholder")
    frame = {
        "c2w": np.eye(4).reshape(-1).tolist(),
        "fx": 100.0,
        "fy": 100.0,
        "cx": 50.0,
        "cy": 50.0,
        "car_pos": [0.0, 0.0, 0.0],
    }
    payload = {"frames": [frame, {**frame, "car_pos": [1.0, 0.0, 0.0]}]}
    if mass is not None:
        payload["car_mass_kg"] = mass
    (path / "gt.json").write_text(json.dumps(payload))


def test_unity_scene_config_covers_all_published_table4_views() -> None:
    config = module.load_config(module.DEFAULT_CONFIG)
    assert len(config["cases"]) == 8
    assert "table4/RacingCar/first" in config["cases"]
    assert "table4/RacingCar/third" in config["cases"]
    assert "table4/BilliardGame/first" in config["cases"]
    assert "table4/BilliardGame/third" in config["cases"]
    billiard_third = config["cases"]["table4/BilliardGame/third"]
    assert "ball_wasd" in billiard_third["runtime_args"]
    assert "300" in billiard_third["runtime_args"]


def test_racing_export_records_measured_mass(tmp_path: Path, monkeypatch) -> None:
    _racing_capture(tmp_path, mass=1000.0)
    video = tmp_path / "unity_groundtruth.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(module, "encode_frames", lambda *_args: video)
    monkeypatch.setattr(
        module.cv2,
        "imread",
        lambda *_args, **_kwargs: np.arange(300, dtype=np.uint8).reshape(10, 10, 3),
    )

    artifacts = module.export_artifacts(tmp_path, "racing", 30.0)

    assert json.loads(
        (tmp_path / "evaluation_artifacts/object_masses.json").read_text()
    ) == {"car": 1000.0}
    assert "object_masses.json" in artifacts
    assert "plane_calibration.json" in artifacts


def test_racing_export_rejects_missing_mass(tmp_path: Path) -> None:
    _racing_capture(tmp_path, mass=None)
    with pytest.raises(ValueError, match="positive car_mass_kg"):
        module.export_artifacts(tmp_path, "racing", 30.0)


def test_racing_export_rejects_non_square_pixel_intrinsics(tmp_path: Path) -> None:
    _racing_capture(tmp_path, mass=1000.0)
    payload = json.loads((tmp_path / "gt.json").read_text())
    for frame in payload["frames"]:
        frame["fx"] = 16.0 / 9.0 * frame["fy"]
    (tmp_path / "gt.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="fx == fy"):
        module.export_artifacts(tmp_path, "racing", 30.0)


def test_partial_capture_status_preserves_untouched_cells(tmp_path: Path) -> None:
    config = {"cases": {"a": {}, "b": {}}}
    module.write_capture_status(
        tmp_path, config, [{"case_id": "a", "status": "ready", "version": 1}]
    )
    module.write_capture_status(
        tmp_path, config, [{"case_id": "b", "status": "ready", "version": 2}]
    )

    receipt = json.loads((tmp_path / "capture_status.json").read_text())
    assert receipt["complete_for_config"] is True
    assert receipt["case_count"] == 2
    assert [(row["case_id"], row["version"]) for row in receipt["cases"]] == [
        ("a", 1),
        ("b", 2),
    ]
