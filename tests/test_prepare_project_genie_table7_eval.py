from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.prepare_project_genie_table7_eval import (
    GARDEN_DEPTH_MODEL,
    PAPER_ANCHORS,
    _resample_reference,
    full_turn_source_indices,
    perturbation_source_indices,
    prepare_evaluation,
    sha256,
)


def test_resample_reference_rejects_non_square_pixel_unity_intrinsics() -> None:
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    intrinsics = np.tile([160.0, 90.0, 80.0, 45.0], (2, 1))

    with pytest.raises(ValueError, match="fx == fy"):
        _resample_reference(
            poses=poses,
            intrinsics=intrinsics,
            frame_count=2,
            candidate_width=160,
            candidate_height=90,
            reference_width=160,
            reference_height=90,
        )


def _make_fixture(tmp_path: Path, *, candidate_frame_hash: str = "same") -> tuple[Path, Path]:
    unity_root = tmp_path / "unity"
    unity_root.mkdir()
    frame = unity_root / "frame.png"
    frame.write_bytes(b"frame")
    actual_frame_hash = sha256(frame)
    poses = unity_root / "poses.npy"
    intrinsics = unity_root / "intrinsics.npy"
    np.save(poses, np.stack([np.eye(4)] * 4))
    np.save(intrinsics, np.tile([100.0, 100.0, 50.0, 50.0], (4, 1)))
    transforms = unity_root / "objects.json"
    transforms.write_text(
        json.dumps(
            {
                "objects": {
                    label: {
                        "frames": list(range(4)),
                        "positions_world": [[index, 0, 1] for index in range(4)],
                    }
                    for label in PAPER_ANCHORS
                }
            }
        )
    )
    reference = unity_root / "reference.mp4"
    reference.write_bytes(b"reference")
    unity_receipt = unity_root / "receipt.json"
    unity_receipt.write_text(
        json.dumps(
            {
                "status": "ok",
                "frame_count": 4,
                "capture_fps": 2.0,
                "conditioning_frame": str(frame),
                "conditioning_frame_sha256": actual_frame_hash,
                "resolution": [100, 100],
                "evaluation_artifacts": {
                    "poses_npy": {"path": str(poses), "sha256": sha256(poses)},
                    "intrinsics_npy": {"path": str(intrinsics), "sha256": sha256(intrinsics)},
                    "object_transforms_json": {
                        "path": str(transforms),
                        "sha256": sha256(transforms),
                    },
                    "reference_video": {"path": str(reference), "sha256": sha256(reference)},
                },
            }
        )
    )
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"candidate")
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "status": "validated",
                "repeat_ids": ["run0"],
                "runs": [
                    {
                        "repeat_id": "run0",
                        "status": "validated",
                        "first_frame_sha256": (
                            actual_frame_hash if candidate_frame_hash == "same" else candidate_frame_hash
                        ),
                        "evaluation_video": str(video),
                        "evaluation_video_sha256": sha256(video),
                    }
                ],
            }
        )
    )
    return validation, unity_receipt


def test_prepare_evaluation_proves_frame_match_and_resamples_timeline(tmp_path: Path) -> None:
    validation, unity = _make_fixture(tmp_path)
    output = tmp_path / "prepared"
    receipt = prepare_evaluation(
        validation_path=validation,
        unity_receipt_path=unity,
        output_dir=output,
        probe_video=lambda _: {"frame_count": 4, "width": 100, "height": 100, "fps": 2.0},
        standardize_video=lambda _source, destination, _info, **_kwargs: destination.parent.mkdir(
            parents=True, exist_ok=True
        ) or destination.write_bytes(b"standardized"),
    )
    assert receipt["status"] == "ready"
    assert receipt["paper_anchor_labels"] == list(PAPER_ANCHORS)
    assert np.load(output / "ground_truth" / "run0" / "poses.npy").shape == (4, 4, 4)
    scaled = np.load(output / "ground_truth" / "run0" / "intrinsics.npy")
    assert scaled[0].tolist() == pytest.approx([100.0, 100.0, 50.0, 50.0])
    row = json.loads((output / "candidate_manifest.jsonl").read_text())
    assert row["seed"] == "run0"
    assert row["params"]["eoc_max_objects"] == 5
    assert row["params"]["eoc_text_prompt"] == "tree. stone lantern. rock."
    assert row["params"]["eoc_depth_model"] == GARDEN_DEPTH_MODEL
    assert row["params"]["eoc_fixed_anchor_labels"] == list(PAPER_ANCHORS)


def test_prepare_evaluation_rejects_mismatched_conditioning_frame(tmp_path: Path) -> None:
    validation, unity = _make_fixture(tmp_path, candidate_frame_hash="wrong")
    with pytest.raises(ValueError, match="conditioning frame does not match"):
        prepare_evaluation(
            validation_path=validation,
            unity_receipt_path=unity,
            output_dir=tmp_path / "prepared",
            probe_video=lambda _: {"frame_count": 7, "width": 100, "height": 100, "fps": 24.0},
            standardize_video=lambda *_args, **_kwargs: None,
        )


def test_full_turn_mapping_respects_periodic_direction_and_start_frame() -> None:
    assert full_turn_source_indices(
        source_count=4, output_count=4, start_frame=0, target_yaw_deg=-360.0
    ) == [0, 3, 2, 1]
    assert full_turn_source_indices(
        source_count=4, output_count=4, start_frame=2, target_yaw_deg=360.0
    ) == [2, 3, 0, 1]


def test_perturbation_mapping_follows_measured_action_then_holds() -> None:
    indices, timing = perturbation_source_indices(
        source_count=8,
        output_count=8,
        start_frame=0,
        target_yaw_deg=-90.0,
        action_trace=[
            {
                "target_yaw_deg": -90.0,
                "accepted_at_sec": 1.0,
                "completed_at_sec": 3.0,
            },
            {"completed_at_sec": 8.0},
        ],
    )
    assert indices == [0, 0, 7, 6, 6, 6, 6, 6]
    assert timing["timeline_end_sec"] == 8.0


def test_prepare_evaluation_marks_browser_capture_for_hud_crop(tmp_path: Path) -> None:
    validation, unity = _make_fixture(tmp_path)
    validation_doc = json.loads(validation.read_text())
    validation_doc["runs"][0]["artifact_status"] = "action_aligned_capture_ready"
    validation.write_text(json.dumps(validation_doc))
    seen: dict[str, str] = {}

    def standardize(_source: Path, destination: Path, _info: dict, **kwargs: object) -> None:
        seen["source_kind"] = str(kwargs["source_kind"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"standardized")

    output = tmp_path / "prepared"
    prepare_evaluation(
        validation_path=validation,
        unity_receipt_path=unity,
        output_dir=output,
        probe_video=lambda path: (
            {"frame_count": 4, "width": 100, "height": 100, "fps": 2.0}
            if path.parent.name == "standardized_videos"
            else {"frame_count": 4, "width": 2446, "height": 1738, "fps": 2.0}
        ),
        standardize_video=standardize,
    )
    assert seen["source_kind"] == "browser_capture"
