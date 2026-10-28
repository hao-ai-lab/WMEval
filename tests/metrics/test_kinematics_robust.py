"""Candidate-camera lifting and robust mechanics (evaluation/modules/kinematics_robust.py).

Hermetic tests render a textured ground plane from a known moving camera and
use synthetic metric tracks; data-dependent checks on the Lite references are
marked ``integration``.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from evaluation.modules import kinematics_robust as kr
from evaluation.modules.kinematics_mechanics import (
    _map_pixels_to_unity_plane_by_pose,
    extended_kinematics_from_tracks,
)

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "lite"
W, H = 320, 180
FX = FY = 260.0
CX, CY = W / 2.0, H / 2.0
K = np.asarray([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])


def _c2w(x: float, z: float, yaw_deg: float, height: float = 3.0, pitch_deg: float = 20.0) -> np.ndarray:
    """Camera-to-world pose in the module's convention (camera y down, z forward)."""
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    forward = np.asarray([np.sin(yaw) * np.cos(pitch), -np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    right = np.asarray([np.cos(yaw), 0.0, -np.sin(yaw)])
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2] = right, down, forward
    pose[:3, 3] = [x, height, z]
    return pose


def _image_from_world_xz(pose: np.ndarray) -> np.ndarray:
    rot, centre = pose[:3, :3], pose[:3, 3]
    return K @ rot.T @ np.stack([[1.0, 0, 0], [0, 0, 1.0], -centre], axis=1)


def _render(poses, texture: np.ndarray, texel_m: float, origin_xz) -> list:
    world_from_texture = np.asarray(
        [[texel_m, 0, origin_xz[0]], [0, texel_m, origin_xz[1]], [0, 0, 1.0]]
    )
    vs, us = np.mgrid[0:H, 0:W]
    frames = []
    for pose in poses:
        image = cv2.warpPerspective(
            texture, _image_from_world_xz(pose) @ world_from_texture, (W, H), flags=cv2.INTER_LINEAR
        )
        ray_y = (pose[:3, :3] @ np.stack([(us - CX) / FX, (vs - CY) / FY, np.ones_like(us, float)]).reshape(3, -1))[1]
        image[(ray_y.reshape(H, W) >= -1e-3)] = 128  # sky
        frames.append(image)
    return frames


@pytest.fixture(scope="module")
def drive():
    rng = np.random.default_rng(0)
    texel_m = 0.03
    texture = rng.integers(0, 255, size=(2400, 1600), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (0, 0), 1.5)
    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX)
    poses, x, z, yaw = [], 0.0, 0.0, 0.0
    for t in range(45):
        poses.append(_c2w(x, z, yaw))
        speed = 0.25 if t < 30 else 0.1
        yaw += 0.6 if 10 <= t < 30 else 0.0
        x += speed * np.sin(np.radians(yaw))
        z += speed * np.cos(np.radians(yaw))
    frames = _render(poses, texture, texel_m, origin_xz=(-24.0, -6.0))
    return {"poses": np.stack(poses), "frames": frames}


def test_unity_camera_motion_classification(drive):
    assert kr.unity_camera_motion(drive["poses"])["moving"] is True
    static = np.repeat(drive["poses"][:1], 20, axis=0)
    info = kr.unity_camera_motion(static)
    assert info["moving"] is False and info["translation_std_m"] == 0.0


def test_fit_rigid_2d_recovers_known_motion():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(50, 2)) * 5
    angle = np.radians(12.0)
    rot = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    target = source @ rot.T + np.asarray([1.5, -0.7])
    transform, scale = kr.fit_rigid_2d(source, target)
    np.testing.assert_allclose(transform[:2, :2], rot, atol=1e-9)
    np.testing.assert_allclose(transform[:2, 2], [1.5, -0.7], atol=1e-9)
    assert scale == pytest.approx(1.0, abs=1e-9)


def test_ground_side_mask_excludes_sky_and_margin():
    h0 = np.linalg.inv(_image_from_world_xz(_c2w(0.0, 0.0, 0.0, pitch_deg=10.0)))
    mask = kr.ground_side_mask(h0, (H, W), margin_frac=0.05)
    # horizon of a camera pitched 10 deg down: v = cy - fy*tan(10 deg)
    horizon = CY - FY * np.tan(np.radians(10.0))
    assert mask[int(horizon) - 2].max() == 0  # sky
    assert mask[int(horizon + 0.05 * H) - 1].max() == 0  # margin below the horizon
    assert mask[H - 20, W // 2] == 255  # near ground


@pytest.mark.parametrize("model", ["ground_rigid", "homography"])
def test_plane_homography_chain_recovers_camera_motion(drive, model):
    poses = drive["poses"]
    h0 = np.linalg.inv(_image_from_world_xz(poses[0]))
    transforms, diag = kr.estimate_plane_camera_motion(
        drive["frames"], h0, fps=30.0, params={"motion_model": model, "keyframe_interval_sec": 0.3}
    )
    assert diag["coverage"] == 1.0 and diag["failed_frame_count"] == 0
    assert diag["keyframe_count"] >= 4  # re-anchored, not one long track
    grid = np.asarray([[u, v] for u in range(40, W - 40, 40) for v in range(H // 2 + 20, H - 10, 20)], float)
    path = float(np.sum(np.linalg.norm(np.diff(poses[:, [0, 2], 3], axis=0), axis=1)))
    errors = []
    for frame, g in transforms.items():
        truth = kr.apply_homography(np.linalg.inv(_image_from_world_xz(poses[frame])), grid)
        errors.append(np.max(np.linalg.norm(kr.apply_homography(g, grid) - truth, axis=1)))
    assert path > 8.0
    # the free 8-DOF chain drifts even on a perfect plane; the rigid default does not
    tolerance = 0.02 if model == "ground_rigid" else 0.3
    assert max(errors) < tolerance * path, (max(errors), path)
    if model == "ground_rigid":
        centres = kr.camera_centres_from_plane_homographies(
            transforms, np.tile([FX, FY, CX, CY], (len(poses), 1)), 0.0
        )
        final = max(centres)
        np.testing.assert_allclose(centres[final][[0, 2]], poses[final][[0, 2], 3], atol=0.02 * path)
        assert kr.ego_reliability(diag)[0] is True


def test_lift_points_maps_contact_pixels(drive):
    poses = drive["poses"]
    transforms = {f: np.linalg.inv(_image_from_world_xz(p)) for f, p in enumerate(poses)}
    world = np.asarray([0.3, 12.0])
    pixels = {
        f: kr.apply_homography(_image_from_world_xz(p), world)[0] for f, p in enumerate(poses)
    }
    lifted = kr.lift_points(pixels, transforms)
    for point in lifted.values():
        np.testing.assert_allclose(point, world, atol=1e-6)


def _speed_profile_track(n=150, fps=30.0, accel=4.0, top=6.0, kink_frame=None, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    speed = np.minimum(accel * np.maximum(t - 0.5, 0.0), top)
    s = np.cumsum(speed) / fps
    heading = np.zeros(n)
    if kink_frame is not None:
        heading[kink_frame:] = np.radians(60.0)
    # integrate with heading
    vel = np.stack([speed * np.sin(heading), speed * np.cos(heading)], axis=1)
    pos = np.cumsum(vel, axis=0) / fps
    pos += rng.normal(scale=noise, size=pos.shape)
    return {int(f): pos[f] for f in range(n)}, s


def test_robust_mechanics_self_small_and_negative_large():
    gt, _ = _speed_profile_track()
    noisy, _ = _speed_profile_track(noise=0.01, seed=3)  # 1 cm tracking jitter
    negative, _ = _speed_profile_track(accel=2.0, top=3.5)
    self_row = kr.robust_mechanics(noisy, gt, 30.0, mass_kg=2.0)
    neg_row = kr.robust_mechanics(negative, gt, 30.0, mass_kg=2.0)
    assert self_row["status"] == "computed"
    assert self_row["robust_error_rate_pct"] < 5.0
    assert self_row["peak_speed_error_pct"] < 5.0
    assert neg_row["robust_error_rate_pct"] > 25.0
    assert neg_row["peak_speed_error_pct"] > 25.0
    # the legacy one-step finite difference on the same jitter is far noisier
    frames = sorted(noisy)
    one_step = np.asarray([np.linalg.norm(noisy[f] - noisy[f - 1]) * 30.0 for f in frames[1:]])
    truth = np.asarray([np.linalg.norm(gt[f] - gt[f - 1]) * 30.0 for f in frames[1:]])
    moving = truth > 0.2 * truth.max()
    legacy_like = 100.0 * np.mean(np.abs(one_step[moving] - truth[moving]) / truth[moving])
    assert legacy_like > 2.0 * self_row["robust_momentum_error_pct"]


def test_robust_mechanics_contact_momentum_change():
    gt, _ = _speed_profile_track(n=120, accel=40.0, top=4.0, kink_frame=70)
    events = kr.detect_contact_events(gt, 30.0, peak_speed=4.0)
    assert any(abs(event - 70) <= 2 for event in events)  # the deflection (the launch is one too)
    same = kr.robust_mechanics(gt, gt, 30.0, mass_kg=0.17)
    assert same["contact_dp_error_pct"] == pytest.approx(0.0, abs=1e-6)
    straight, _ = _speed_profile_track(n=120, accel=40.0, top=4.0)  # no deflection
    wrong = kr.robust_mechanics(straight, gt, 30.0, mass_kg=0.17)
    deflection = [row for row in wrong["contact_events"] if abs(row["frame"] - 70) <= 2]
    assert deflection and deflection[0]["dp_error_pct"] > 80.0
    assert wrong["contact_dp_error_pct"] >= 40.0  # mean with the (correct) launch event


def test_robust_mechanics_never_moving_object():
    still = {f: np.asarray([1.0, 2.0]) for f in range(60)}
    assert kr.robust_mechanics(still, still, 30.0, mass_kg=1.0)["status"] == "object_never_moves"


def test_extended_kinematics_static_camera_equals_legacy_lifting():
    pose = _c2w(0.0, 0.0, 0.0)
    poses = np.repeat(pose[None], 60, axis=0)
    intrinsics = np.tile([FX, FY, CX, CY], (60, 1))
    world = {f: np.asarray([0.02 * f, 4.0 + 0.05 * f]) for f in range(60)}
    pixels = {f: kr.apply_homography(_image_from_world_xz(pose), w)[0] for f, w in world.items()}
    trajectories = {
        "ball": [{"frame": f, "centroid_uv": list(p), "bbox_bottom_center_uv": list(p)} for f, p in pixels.items()]
    }
    gt = {"objects": {"ball": {"frames": list(world), "positions_world": [[w[0], 0.0, w[1] + 0.05] for w in world.values()]}}}
    calibration = {"plane": {"value_m": 0.0}, "anchors": [], "image_size": [W, H]}
    out = extended_kinematics_from_tracks(
        trajectories, fps=30.0, track_point="bbox_bottom_center", video_path=None, gt=gt,
        poses=poses, intrinsics=intrinsics, plane_calibration=calibration, masses={"ball": 0.2},
        required_labels=["ball"], alignment_tolerance_frames=5,
    )
    assert out["lifting"] == "unity_camera_static" and out["reliable"] is True
    legacy = _map_pixels_to_unity_plane_by_pose(pixels, poses, intrinsics, 0.0)
    assert out["summary"]["fde_m_ego_mean"] == pytest.approx(
        float(np.linalg.norm(legacy[59] - np.asarray([world[59][0], world[59][1] + 0.05]))), abs=1e-6
    )
    assert out["summary"]["fde_m_ego_mean"] == pytest.approx(0.05, abs=1e-6)
    assert out["summary"]["robust_error_rate_pct"] == pytest.approx(0.0, abs=1e-3)


def test_aggregate_exposes_extended_metrics():
    from evaluation.lite.aggregate import extract_metrics
    from evaluation.lite.suite_loader import load_suite

    suite = load_suite(SUITE, check_case_dirs=False)
    dimension = suite.dimensions["kinematics_mechanics"]
    results = {
        "kinematics_mechanics": {
            "trajectory_coverage": {"status": "complete", "missing_required_label_count": 0},
            "extended_kinematics": {
                "summary": {
                    "lifting": "candidate_camera_ground_plane",
                    "reliable": True,
                    "fde_m_ego_mean": 0.4,
                    "dtw_m_ego_mean": 0.2,
                    "robust_error_rate_pct": 3.0,
                    "peak_speed_error_pct": 1.0,
                    "contact_dp_error_pct": None,
                }
            },
        }
    }
    metrics = extract_metrics(dimension, results)
    assert metrics["fde_m_ego_mean"] == 0.4 and metrics["dtw_m_ego_mean"] == 0.2
    assert metrics["robust_error_rate_pct"] == 3.0 and metrics["contact_dp_error_pct"] is None
    assert metrics["ego_lift_reliable"] == 1.0 and metrics["ego_camera_moving"] == 1.0
    for name in ("fde_m_ego_mean", "dtw_m_ego_mean", "robust_error_rate_pct"):
        assert name in dimension.aggregate.extra_means
    assert dimension.aggregate.headline == "fde_m_mean"  # headline unchanged


# ---------------------------------------------------------------------------
# Data-dependent checks on the Lite references (perfect tracker = Unity GT
# projected through the Unity camera, so only the camera estimate is tested)
# ---------------------------------------------------------------------------
def _projected_track(artifacts: Path, label: str):
    from evaluation.modules.kinematics_mechanics import _project_unity_trajectory_to_pixels

    gt = json.loads((artifacts / "object_transforms.json").read_text())
    entry = {k.lower(): v for k, v in gt["objects"].items()}[label]
    world = [(int(f), np.asarray(p, float)) for f, p in zip(entry["frames"], entry["positions_world"])]
    pixels = _project_unity_trajectory_to_pixels(world, np.load(artifacts / "poses.npy"), np.load(artifacts / "intrinsics.npy"))
    # a detector-sized box around the projected origin (masks the object and
    # its shadow out of the camera-motion features, as tracked boxes do)
    points = [
        {
            "frame": f,
            "centroid_uv": list(p),
            "bbox_bottom_center_uv": list(p),
            "bbox_xyxy": [p[0] - 90.0, p[1] - 80.0, p[0] + 90.0, p[1] + 40.0],
        }
        for f, p in sorted(pixels.items())
    ]
    return gt, points


def _ego_fde(case_dir: Path, gt_dir: Path, label: str) -> dict:
    artifacts = case_dir / "reference" / "artifacts"
    gt_artifacts = gt_dir / "reference" / "artifacts"
    _, points = _projected_track(artifacts, label)
    gt, _ = _projected_track(gt_artifacts, label)
    calibration = json.loads((gt_artifacts / "plane_calibration.json").read_text())
    return extended_kinematics_from_tracks(
        {label: points}, fps=30.0, track_point="centroid", video_path=str(case_dir / "reference" / "reference.mp4"),
        gt=gt, poses=np.load(gt_artifacts / "poses.npy"), intrinsics=np.load(gt_artifacts / "intrinsics.npy"),
        plane_calibration=calibration, masses={label: 1000.0}, required_labels=[label], alignment_tolerance_frames=5,
    )


EGO = SUITE / "synthetic" / "ego_vehicle" / "RacingCar__kinematics_third"


@pytest.mark.integration("synthetic_data")
def test_ego_vehicle_self_match_below_negative():
    self_row = _ego_fde(EGO / "s03_drive_brake", EGO / "s03_drive_brake", "car")
    negative = _ego_fde(EGO / "s20_drive_coast_brake", EGO / "s03_drive_brake", "car")
    assert self_row["lifting"] == "candidate_camera_ground_plane"
    raw_self = self_row["labels"]["car"]["fde_m_ego"]
    raw_negative = negative["labels"]["car"]["fde_m_ego"]
    assert raw_self < 1.0
    assert raw_negative > 10.0 * raw_self


@pytest.mark.integration("case_data")
def test_racingcar_reference_camera_path_matches_unity():
    case = SUITE / "cases" / "RacingCar" / "kinematics_third"
    artifacts = case / "reference" / "artifacts"
    calibration = json.loads((artifacts / "plane_calibration.json").read_text())
    poses = np.load(artifacts / "poses.npy")
    h0, rms = kr.frame0_plane_homography(calibration, (1280, 720))
    assert rms < 1e-3
    _, points = _projected_track(artifacts, "car")
    boxes = kr.object_boxes_from_trajectory(points)
    transforms, diag = kr.estimate_plane_camera_motion(
        str(case / "reference" / "reference.mp4"), h0, fps=30.0, object_boxes=boxes
    )
    c0 = poses[0][[0, 2], 3]
    errors = [
        np.linalg.norm(kr.apply_homography(g @ np.linalg.inv(h0), c0)[0] - poses[f][[0, 2], 3])
        for f, g in transforms.items()
    ]
    assert diag["coverage"] == 1.0
    assert errors[150] < 1.0  # before the barrier crash
    assert errors[-1] < 4.0  # 35 m path incl. a crash


def test_robust_mechanics_onset_alignment_and_detector_gap():
    gt, _ = _speed_profile_track()
    delayed = {f + 3: p for f, p in gt.items() if f + 3 < 150}  # 0.1 s action latency
    row = kr.robust_mechanics(delayed, gt, 30.0, mass_kg=1.0, alignment_tolerance_frames=5)
    assert row["temporal_offset_frames"] == 3
    assert row["robust_error_rate_pct"] < 1.0
    # a detector gap across the launch is not mistaken for a late onset
    gappy = {f: p for f, p in gt.items() if not 14 <= f <= 24}
    row = kr.robust_mechanics(gappy, gt, 30.0, mass_kg=1.0, alignment_tolerance_frames=5)
    # The onset sits on the speed-threshold boundary, so float differences across
    # platforms can move it by one frame; an 11-frame gap read as a late onset would
    # shift it by several frames.
    assert abs(row["temporal_offset_frames"]) <= 1
    assert row["robust_error_rate_pct"] < 1.0


def test_object_camera_attachment_detects_chase_camera():
    poses, world = [], []
    for f in range(60):
        z = 0.3 * f
        poses.append(_c2w(0.0, z, 0.0))
        world.append((f, np.asarray([0.0, 0.0, z + 8.0])))  # carried 8 m ahead
    intrinsics = np.tile([FX, FY, CX, CY], (60, 1))
    assert kr.object_camera_attachment(world, np.stack(poses), intrinsics, W)["attached"] is True
    rolling = [(f, np.asarray([0.05 * f, 0.0, 8.0])) for f in range(60)]  # crosses the view
    static_cam = np.repeat(poses[:1], 60, axis=0)
    assert kr.object_camera_attachment(rolling, static_cam, intrinsics, W)["attached"] is False
