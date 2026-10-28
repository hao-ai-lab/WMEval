from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_kinematics_mechanics_registered():
    from evaluation.registry import MODULE_REGISTRY, get_available_modules

    assert "kinematics_mechanics" in MODULE_REGISTRY
    assert "kinematics_mechanics" in get_available_modules()


def test_fde_and_dtw():
    from evaluation.modules.kinematics_mechanics import _dtw, _dtw_mean, _fde

    traj_hat = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    traj_ref = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])

    assert _fde(traj_hat, traj_ref) == pytest.approx(1.0)
    assert _dtw(traj_hat, traj_ref) == pytest.approx(1.0)
    assert _dtw_mean(traj_hat, traj_ref) == pytest.approx(1.0 / 3.0)

    # Identical trajectories: zero error.
    assert _fde(traj_hat, traj_hat) == pytest.approx(0.0)
    assert _dtw(traj_hat, traj_hat) == pytest.approx(0.0)


def test_map_pixels_to_metric_plane():
    from evaluation.modules.kinematics_mechanics import _map_pixels_to_plane

    # x = 0.01 u, z = 0.02 v
    homography = np.array([[0.01, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 1.0]])
    mapped = _map_pixels_to_plane(
        {0: np.array([100.0, 50.0]), 1: np.array([200.0, 25.0])},
        {0: homography, 1: homography},
    )

    assert mapped[0] == pytest.approx([1.0, 1.0])
    assert mapped[1] == pytest.approx([2.0, 0.5])


def test_map_pixels_to_unity_plane_by_pose():
    from evaluation.modules.kinematics_mechanics import _map_pixels_to_unity_plane_by_pose

    # Camera at (0, 1, 0), CV axes right/down/forward. Pixel (50, 150)
    # produces ray (0, 1, 1), intersecting y=0 at world (0, 0, 1).
    pose = np.eye(4)
    pose[1, 1] = -1.0
    pose[1, 3] = 1.0
    mapped = _map_pixels_to_unity_plane_by_pose(
        {0: np.array([50.0, 150.0])},
        np.stack([pose]),
        np.array([[100.0, 100.0, 50.0, 50.0]]),
        0.0,
    )

    assert mapped[0] == pytest.approx([0.0, 1.0])


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
    # No anchors configured: no actual compensation math runs, but downstream
    # code always reads "centroid_compensated", so it must default to the raw
    # centroid rather than being omitted (regression test for a real crash:
    # KeyError when kin_static_anchor_labels is left empty).
    assert compensated[1][0]["centroid_compensated"] == [1.0, 2.0]
    assert compensated[1][0]["centroid"] == [1.0, 2.0]


def test_round_sig2():
    from evaluation.modules.kinematics_mechanics import _round_sig2

    assert _round_sig2(123.456) == pytest.approx(120.0)
    assert _round_sig2(0.0034521) == pytest.approx(0.0035)
    assert _round_sig2(0.0) == 0.0


def test_linear_velocity_fit_recovers_known_slope():
    from evaluation.modules.kinematics_mechanics import _linear_velocity_fit

    fps = 10.0
    frames = [0, 1, 2, 3, 4, 5]
    true_velocity = np.array([2.0, -1.0, 0.5])
    points = np.array([true_velocity * (f / fps) for f in frames])

    v_hat = _linear_velocity_fit(frames, points, fps)

    assert v_hat == pytest.approx(true_velocity, abs=1e-9)


def test_linear_velocity_fit_two_points_falls_back_to_finite_difference():
    from evaluation.modules.kinematics_mechanics import _linear_velocity_fit

    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    v_hat = _linear_velocity_fit([0, 5], points, fps=10.0)

    # (x1 - x0) / ((5 - 0) / 10) = (1,2,3) / 0.5
    assert v_hat == pytest.approx([2.0, 4.0, 6.0])


def test_linear_velocity_fit_smooths_noise_better_than_two_point_diff():
    from evaluation.modules.kinematics_mechanics import _linear_velocity_fit

    fps = 10.0
    frames = list(range(6))
    true_velocity = np.array([1.0, 0.0, 0.0])
    rng = np.random.default_rng(0)
    clean = np.array([true_velocity * (f / fps) for f in frames])
    noisy = clean.copy()
    # Perturb only the endpoints heavily -- a 2-point finite difference
    # would be fully corrupted by this; a multi-point linear fit should
    # stay close to the true slope since it uses every point.
    noisy[0] += [0.5, 0.0, 0.0]
    noisy[-1] += [-0.5, 0.0, 0.0]

    fit_velocity = _linear_velocity_fit(frames, noisy, fps)
    two_point_velocity = (noisy[-1] - noisy[0]) / ((frames[-1] - frames[0]) / fps)

    fit_error = float(np.linalg.norm(fit_velocity - true_velocity))
    two_point_error = float(np.linalg.norm(two_point_velocity - true_velocity))
    assert fit_error < two_point_error


def test_select_sustained_peak_step_rejects_isolated_tracking_spike():
    from evaluation.modules.kinematics_mechanics import _select_sustained_peak_step

    points = {
        0: np.array([0.0, 0.0]),
        1: np.array([1.0, 0.0]),
        2: np.array([2.0, 0.0]),
        3: np.array([102.0, 0.0]),  # isolated detector jump
        4: np.array([3.0, 0.0]),
        5: np.array([4.0, 0.0]),
    }

    peak = _select_sustained_peak_step(points, fps=1.0, min_neighbor_ratio=0.5)

    assert peak is not None
    assert peak["speed_m_s"] == pytest.approx(1.0)
    assert peak["frame"] == 1


def test_aligned_peak_event_handles_small_physical_event_offset():
    from evaluation.modules.kinematics_mechanics import _aligned_peak_event_stage

    # Same three-frame 2 m/s event, with the candidate occurring four frames
    # earlier. Shared-frame windowing would mix candidate motion with Unity
    # rest, while bounded event alignment compares the corresponding event.
    ground_truth = {frame: np.array([0.0, 0.0]) for frame in range(20)}
    candidate = {frame: np.array([0.0, 0.0]) for frame in range(20)}
    for frame in range(11, 20):
        ground_truth[frame] = np.array([2.0 * (frame - 10), 0.0])
    for frame in range(7, 20):
        candidate[frame] = np.array([2.0 * (frame - 6), 0.0])

    stage = _aligned_peak_event_stage(
        "ball",
        1.0,
        candidate,
        ground_truth,
        1.0,
        tolerance_frames=5,
        min_neighbor_ratio=0.5,
    )

    assert stage is not None
    assert stage["estimator"] == "aligned_peak_finite_difference"
    assert stage["temporal_offset_frames"] == -4
    assert stage["momentum_kg_m_s"]["error_rate_pct"] == 0.0
    assert stage["kinetic_energy_j"]["error_rate_pct"] == 0.0


def test_mechanical_quantities_exact_match_paper_formula_explodes_near_zero():
    from evaluation.modules.kinematics_mechanics import _mechanical_quantities_exact_match

    # Real absolute error is small (0.02), but Eq 13's default denominator
    # (|q_gt|) is near-zero, so the paper's own formula explodes -- this
    # locks in that the formula is intentionally left unmodified (paper
    # fidelity), not "fixed" by changing what error_rate_pct means.
    _, error_rate_pct, diag = _mechanical_quantities_exact_match(
        {"ball": 0.02}, {"ball": 0.0001}
    )
    assert error_rate_pct > 1000.0
    assert diag["absolute_errors"]["ball"] == pytest.approx(0.0199, abs=1e-4)


def test_mechanical_quantities_exact_match_near_zero_threshold_flags_label():
    from evaluation.modules.kinematics_mechanics import _mechanical_quantities_exact_match

    _, _, diag = _mechanical_quantities_exact_match(
        {"ball": 0.02}, {"ball": 0.0001}, near_zero_threshold={"ball": 0.001}
    )
    assert diag["near_zero_ground_truth_baseline_labels"] == ["ball"]

    _, _, diag_not_near_zero = _mechanical_quantities_exact_match(
        {"ball": 1.02}, {"ball": 1.0}, near_zero_threshold={"ball": 0.001}
    )
    assert "near_zero_ground_truth_baseline_labels" not in diag_not_near_zero


def test_mechanical_quantities_exact_match_reference_scale_denominator():
    from evaluation.modules.kinematics_mechanics import _mechanical_quantities_exact_match

    # Same absolute error as the near-zero-baseline case above, but with a
    # real reference-scale denominator (e.g. this object's own peak
    # momentum elsewhere in its rollout) instead of the near-zero
    # instantaneous q_gt -- this is the actual fix: a sane, bounded
    # percentage instead of Eq 13's default explosion.
    _, error_rate_pct, _ = _mechanical_quantities_exact_match(
        {"ball": 0.02}, {"ball": 0.0001}, denominator={"ball": 1.0}
    )
    assert error_rate_pct == pytest.approx(1.99, abs=0.01)


def test_metric_plane_mechanics_and_table_summary_use_peak_motion():
    from evaluation.modules.kinematics_mechanics import (
        _mechanics_values_from_points,
        _summarize_mechanical_quantities,
    )

    frames = [8, 9, 10]
    gt = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    estimated = gt.copy()
    stage = _mechanics_values_from_points(
        "car", 1000.0, frames, estimated, gt, 1.0,
        ref_momentum=1000.0, ref_energy=500.0, peak_speed=1.0,
    )
    summary = _summarize_mechanical_quantities(
        {
            "car": {
                "stages": {"terminal": stage, "peak_motion": stage},
            }
        }
    )
    assert summary is not None
    assert summary["selected_stage_by_label"] == {"car": "peak_motion"}
    assert summary["quantity_count"] == 2
    assert summary["exact_match_pct"] == 100.0
    assert summary["error_rate_pct"] == 0.0


def test_table_summary_prefers_aligned_peak_event():
    from evaluation.modules.kinematics_mechanics import (
        _mechanics_values_from_points,
        _summarize_mechanical_quantities,
    )

    frames = [0, 1]
    gt = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    bad = np.asarray([[0.0, 0.0], [2.0, 0.0]])
    legacy = _mechanics_values_from_points(
        "car", 1.0, frames, bad, gt, 1.0,
        ref_momentum=1.0, ref_energy=0.5, peak_speed=1.0,
    )
    aligned = _mechanics_values_from_points(
        "car", 1.0, frames, gt, gt, 1.0,
        ref_momentum=1.0, ref_energy=0.5, peak_speed=1.0,
    )
    aligned["estimator"] = "aligned_peak_finite_difference"
    summary = _summarize_mechanical_quantities(
        {"car": {"stages": {"peak_motion": legacy, "aligned_peak_event": aligned}}}
    )

    assert summary is not None
    assert summary["selected_stage_by_label"] == {"car": "aligned_peak_event"}
    assert summary["error_rate_pct"] == 0.0


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
