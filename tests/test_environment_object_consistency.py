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


def test_no_character_sentinel_disables_detector_branch():
    from evaluation.modules.environment_object_consistency import (
        _character_prompt_is_disabled,
    )

    assert _character_prompt_is_disabled("")
    assert _character_prompt_is_disabled("no_character_configured_for_this_case_xyz.")
    assert not _character_prompt_is_disabled("the white cue ball.")


def test_pipe_delimited_prompts_run_independent_detector_calls(monkeypatch):
    import evaluation.modules.environment_object_consistency as module

    calls = []

    def fake_detector(frame, prompt, model_id, box_threshold, text_threshold, device):
        calls.append(prompt)
        offset = float(len(calls))
        return (
            np.asarray([[offset, 0.0, offset + 1.0, 1.0]], dtype=np.float32),
            np.asarray([0.9], dtype=np.float32),
            [prompt.rstrip(".")],
        )

    monkeypatch.setattr(module, "_detect_objects_groundingdino", fake_detector)
    boxes, scores, labels = module._detect_prompt_groups(
        np.zeros((8, 8, 3), dtype=np.uint8),
        "yellow ball.|red ball.",
        "model",
        0.3,
        0.25,
        "cpu",
    )
    assert calls == ["yellow ball.", "red ball."]
    assert boxes.shape == (2, 4)
    assert scores.tolist() == pytest.approx([0.9, 0.9])
    assert labels == ["yellow ball", "red ball"]


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


def test_fixed_anchor_assignment_uses_unity_projection_and_canonical_labels():
    from evaluation.modules.environment_object_consistency import (
        _assign_fixed_anchor_detections,
    )

    anchors = ["Main tree", "left lamp post", "right lamp post"]
    gt = {
        "objects": {
            "Main tree": {"frames": [0], "positions_world": [[0.0, 0.0, 10.0]]},
            "left lamp post": {"frames": [0], "positions_world": [[-2.0, 0.0, 10.0]]},
            "right lamp post": {"frames": [0], "positions_world": [[2.0, 0.0, 10.0]]},
        }
    }
    poses = np.stack([np.eye(4)])
    intrinsics = np.asarray([[100.0, 100.0, 50.0, 50.0]])
    # Expected centers are tree=50, left=30, right=70. Detector labels are
    # intentionally generic and duplicated; analytical projection supplies
    # the identity that text labels cannot.
    boxes = np.asarray(
        # The tree intentionally covers >60% of the frame: close landmarks
        # must survive fixed-anchor assignment even though the generic seed
        # deduper rejects boxes this large as potentially degenerate.
        [[5, 15, 95, 85], [25, 40, 35, 60], [65, 40, 75, 60]],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    labels = ["tree", "lamp post", "lamp post"]

    assigned = _assign_fixed_anchor_detections(
        boxes=boxes,
        scores=scores,
        labels=labels,
        fixed_anchor_labels=anchors,
        gt=gt,
        poses=poses,
        intrinsics=intrinsics,
        frame_idx=0,
        frame_width=100,
        frame_height=100,
    )

    assert {entry["label"] for entry in assigned} == set(anchors)
    assert all(entry["assignment_distance_normalized"] == pytest.approx(0.0) for entry in assigned)


def test_fixed_anchor_assignment_supports_repeated_sphere_instances():
    from evaluation.modules.environment_object_consistency import (
        _assign_fixed_anchor_detections,
    )

    anchors = ["yellow sphere left foreground", "yellow sphere right foreground"]
    gt = {
        "objects": {
            anchors[0]: {"frames": [0], "positions_world": [[-2.0, 0.0, 10.0]]},
            anchors[1]: {"frames": [0], "positions_world": [[2.0, 0.0, 10.0]]},
        }
    }
    assigned = _assign_fixed_anchor_detections(
        boxes=np.asarray([[25, 40, 35, 60], [65, 40, 75, 60]], dtype=np.float32),
        scores=np.asarray([0.8, 0.7], dtype=np.float32),
        labels=["yellow sphere", "yellow sphere"],
        fixed_anchor_labels=anchors,
        gt=gt,
        poses=np.stack([np.eye(4)]),
        intrinsics=np.asarray([[100.0, 100.0, 50.0, 50.0]]),
        frame_idx=0,
        frame_width=100,
        frame_height=100,
    )
    assert [entry["label"] for entry in assigned] == anchors


def test_object_permanence_keeps_fixed_anchor_denominator():
    from evaluation.modules.environment_object_consistency import _object_permanence

    track = {
        "n_frames_total": 2,
        "objects": [
            {"label": "a", "is_character": False},
            {"label": "b", "is_character": False},
            {"label": "c", "is_character": False},
        ],
        "detections_by_frame": [
            {"frame": 0, "objects": [{"label": "a", "score": 0.9}]},
            {"frame": 1, "objects": [{"label": "a", "score": 0.9}, {"label": "b", "score": 0.9}]},
        ],
    }

    assert _object_permanence(track, 0.3) == pytest.approx(50.0)


def test_estimate_depth_via_apparent_size_matches_pinhole_formula():
    from evaluation.modules.environment_object_consistency import (
        _estimate_depth_via_apparent_size,
    )

    fx = 500.0
    real_diameter_m = 2.0
    true_depth = 10.0
    # A sphere of real diameter 2.0m at depth 10.0m, seen through fx=500,
    # has apparent pixel diameter = real_diameter_m * fx / depth = 100px.
    apparent_diam_px = real_diameter_m * fx / true_depth
    half = apparent_diam_px / 2.0
    cx, cy = 320.0, 240.0
    bbox = [cx - half, cy - half, cx + half, cy + half]

    track = {
        "frames": [
            {
                "frame": 0,
                "objects": [
                    {"label": "sphere", "centroid_uv": [cx, cy], "bbox_xyxy": bbox}
                ],
            }
        ]
    }
    intrinsics = np.array([[fx, fx, cx, cy]])

    depth_series, point3d_series = _estimate_depth_via_apparent_size(
        track, {"sphere": real_diameter_m}, intrinsics, (480, 640)
    )

    assert depth_series["sphere"] == pytest.approx([true_depth], rel=1e-6)
    # Centroid is at the principal point, so the unprojected (x, y) should
    # be ~0, leaving z == the recovered depth.
    assert point3d_series["sphere"][0] == pytest.approx([0.0, 0.0, true_depth], abs=1e-6)


def test_estimate_depth_via_apparent_size_skips_labels_without_known_diameter():
    from evaluation.modules.environment_object_consistency import (
        _estimate_depth_via_apparent_size,
    )

    track = {
        "frames": [
            {
                "frame": 0,
                "objects": [
                    {"label": "unknown_object", "centroid_uv": [10.0, 10.0], "bbox_xyxy": [0, 0, 20, 20]}
                ],
            }
        ]
    }
    intrinsics = np.array([[100.0, 100.0, 50.0, 50.0]])

    depth_series, point3d_series = _estimate_depth_via_apparent_size(
        track, {"sphere": 1.0}, intrinsics, (100, 100)
    )

    # "unknown_object" has no entry in real_diameter_m, so it must be
    # silently skipped, not crash or fabricate a depth for it.
    assert depth_series == {"sphere": []}
    assert point3d_series == {"sphere": []}


def test_load_object_diameters_schema_validation(tmp_path):
    from evaluation.modules.environment_object_consistency import _load_object_diameters

    with pytest.raises(FileNotFoundError):
        _load_object_diameters(str(tmp_path / "missing.json"))

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_object_diameters(str(bad))

    good = tmp_path / "good.json"
    good.write_text(json.dumps({"Yellow Sphere": 1.3814}), encoding="utf-8")
    diameters = _load_object_diameters(str(good))
    # Case-insensitive lookup convention, matching _load_object_masses.
    assert diameters["yellow sphere"] == pytest.approx(1.3814)


def test_pairwise_distances_from_point3d_matches_original_inline_logic():
    from evaluation.modules.environment_object_consistency import (
        _pairwise_distances_from_point3d,
    )

    point3d_series = {
        "a": [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])],
        "b": [np.array([3.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0])],
        "c": [],  # no detections -- must be excluded, not crash
    }

    distances = _pairwise_distances_from_point3d(point3d_series, ["a", "b", "c"])

    assert distances[("a", "b")] == pytest.approx(4.0)  # mean of 3.0 and 5.0
    assert ("a", "c") not in distances
    assert ("b", "c") not in distances


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


def test_pairwise_distance_support_flags_thin_anchor_coverage() -> None:
    import numpy as np

    from evaluation.modules.environment_object_consistency import _pairwise_distance_support

    track = {"frames": [{"frame": i} for i in range(10)]}
    point = np.zeros(3)
    full = {"a": [point] * 10, "b": [point] * 10}
    support = _pairwise_distance_support(track, full, [("a", "b")])
    assert support["measurable"] is True and support["paired_frame_coverage"] == 1.0
    thin = {"a": [point] * 10, "b": [point] * 3}
    assert _pairwise_distance_support(track, thin, [("a", "b")])["measurable"] is False
    single = {"a": [point] * 10}
    assert _pairwise_distance_support(track, single, [])["measurable"] is False
