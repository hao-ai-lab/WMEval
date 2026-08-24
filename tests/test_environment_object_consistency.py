from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_environment_object_consistency_registered():
    from evaluation.registry import MODULE_REGISTRY, get_available_modules

    assert "environment_object_consistency" in MODULE_REGISTRY
    assert "environment_object_consistency" in get_available_modules()


def test_unity_object_transforms_schema_validation(tmp_path):
    from evaluation.modules.environment_object_consistency import (
        _load_unity_object_transforms,
    )

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not_objects": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_unity_object_transforms(str(bad))

    with pytest.raises(FileNotFoundError):
        _load_unity_object_transforms(str(tmp_path / "missing.json"))

    good = tmp_path / "good.json"
    good.write_text(
        json.dumps({"objects": {"ball": {"frames": [0], "positions_world": [[0, 0, 1]]}}}),
        encoding="utf-8",
    )
    payload = _load_unity_object_transforms(str(good))
    assert "ball" in payload["objects"]


def test_camera_poses_and_intrinsics_shape_validation(tmp_path):
    from evaluation.modules.environment_object_consistency import (
        _load_camera_intrinsics,
        _load_camera_poses,
    )

    bad_poses = tmp_path / "poses.npy"
    np.save(bad_poses, np.zeros((3, 3)))
    with pytest.raises(ValueError):
        _load_camera_poses(str(bad_poses))

    good_poses = tmp_path / "poses_ok.npy"
    np.save(good_poses, np.stack([np.eye(4)] * 3))
    poses = _load_camera_poses(str(good_poses))
    assert poses.shape == (3, 4, 4)

    bad_intr = tmp_path / "intr.npy"
    np.save(bad_intr, np.zeros((3, 3)))
    with pytest.raises(ValueError):
        _load_camera_intrinsics(str(bad_intr))

    good_intr = tmp_path / "intr_ok.npy"
    np.save(good_intr, np.tile([100.0, 100.0, 50.0, 50.0], (3, 1)))
    intr = _load_camera_intrinsics(str(good_intr))
    assert intr.shape == (3, 4)


def test_unity_ground_truth_depths_and_distances():
    from evaluation.modules.environment_object_consistency import (
        _unity_ground_truth_depths_and_distances,
    )

    gt = {
        "objects": {
            "ball": {"frames": [0, 1], "positions_world": [[0, 0, 5], [0, 0, 5]]},
            "ramp": {"frames": [0, 1], "positions_world": [[2, 0, 5], [2, 0, 5]]},
        }
    }
    poses = np.stack([np.eye(4), np.eye(4)])

    depth_by_label, dist_by_pair, missing = _unity_ground_truth_depths_and_distances(
        ["ball", "ramp", "missing_object"], gt, poses
    )

    assert depth_by_label["ball"] == pytest.approx(5.0)
    assert dist_by_pair[("ball", "ramp")] == pytest.approx(2.0)
    assert missing == ["missing_object"]


def test_analyze_environment_object_consistency_requires_candidate_video():
    from evaluation.context import EvaluationContext, SampledVideoFrames, VideoMetadata
    from evaluation.modules.environment_object_consistency import (
        analyze_environment_object_consistency_from_context,
    )

    meta = VideoMetadata(path="x.mp4", fps=24.0, frame_count=1, width=4, height=4, duration_sec=1.0)
    frames = SampledVideoFrames(metadata=meta, timestamps_sec=[0.0], frame_indices=[0], frames=[])
    context = EvaluationContext(
        reference_video="ref.mp4",
        candidate_video="",
        reference=frames,
        candidate=frames,
        shared_duration_sec=1.0,
        eval_fps=4.0,
        max_pairs=1,
    )
    result = analyze_environment_object_consistency_from_context(context)
    assert result["module"] == "environment_object_consistency"
    assert "error" in result
