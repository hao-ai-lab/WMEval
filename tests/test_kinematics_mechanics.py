from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_kinematics_mechanics_registered():
    from evaluation.registry import MODULE_REGISTRY, get_available_modules

    assert "kinematics_mechanics" in MODULE_REGISTRY
    assert "kinematics_mechanics" in get_available_modules()


def test_fde_and_dtw():
    from evaluation.modules.kinematics_mechanics import _dtw, _fde

    traj_hat = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    traj_ref = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])

    assert _fde(traj_hat, traj_ref) == pytest.approx(1.0)
    assert _dtw(traj_hat, traj_ref) == pytest.approx(1.0)

    # Identical trajectories: zero error.
    assert _fde(traj_hat, traj_hat) == pytest.approx(0.0)
    assert _dtw(traj_hat, traj_hat) == pytest.approx(0.0)


def test_project_unity_trajectory_to_pixels():
    from evaluation.modules.kinematics_mechanics import _project_unity_trajectory_to_pixels

    poses = np.stack([np.eye(4), np.eye(4)])
    intrinsics = np.array([[100.0, 100.0, 50.0, 50.0], [100.0, 100.0, 50.0, 50.0]])
    traj_world = [(0, np.array([0.0, 0.0, 10.0])), (1, np.array([0.0, 0.0, 10.0]))]

    projected = _project_unity_trajectory_to_pixels(traj_world, poses, intrinsics)

    assert projected[0] == pytest.approx([50.0, 50.0])
    assert projected[1] == pytest.approx([50.0, 50.0])


def test_compensate_camera_motion_removes_static_anchor_drift():
    from evaluation.modules.kinematics_mechanics import _compensate_camera_motion

    tracks = {
        1: [
            {"frame": 0, "label": "wall", "score": 0.9, "centroid": [10.0, 10.0]},
            {"frame": 1, "label": "wall", "score": 0.9, "centroid": [15.0, 10.0]},
        ],
        2: [
            {"frame": 0, "label": "ball", "score": 0.9, "centroid": [0.0, 0.0]},
            {"frame": 1, "label": "ball", "score": 0.9, "centroid": [5.0, 0.0]},
        ],
    }

    compensated = _compensate_camera_motion(tracks, ["wall"], 2)

    # The ball moved by the same amount as the "camera drift" implied by the
    # wall anchor, so after compensation it should look static.
    ball_pts = compensated[2]
    assert ball_pts[0]["centroid_compensated"] == pytest.approx([0.0, 0.0])
    assert ball_pts[1]["centroid_compensated"] == pytest.approx([0.0, 0.0])


def test_compensate_camera_motion_noop_without_anchors():
    from evaluation.modules.kinematics_mechanics import _compensate_camera_motion

    tracks = {1: [{"frame": 0, "label": "ball", "score": 0.9, "centroid": [1.0, 2.0]}]}
    compensated = _compensate_camera_motion(tracks, [], 1)
    assert compensated is tracks


def test_round_sig2():
    from evaluation.modules.kinematics_mechanics import _round_sig2

    assert _round_sig2(123.456) == pytest.approx(120.0)
    assert _round_sig2(0.0034521) == pytest.approx(0.0035)
    assert _round_sig2(0.0) == 0.0


def test_analyze_kinematics_mechanics_requires_candidate_video():
    from evaluation.context import EvaluationContext, SampledVideoFrames, VideoMetadata
    from evaluation.modules.kinematics_mechanics import (
        analyze_kinematics_mechanics_from_context,
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
    result = analyze_kinematics_mechanics_from_context(context)
    assert result["module"] == "kinematics_mechanics"
    assert "error" in result
