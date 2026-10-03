from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "audit_table4_action_coverage.py"
SPEC = importlib.util.spec_from_file_location("audit_table4_action_coverage", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_table4_action_audit_covers_active_views_and_reference_timelines() -> None:
    report = MODULE.build()

    assert report["table4_case_count"] == 2
    assert report["required_generation_count"] == 10
    assert report["status"] == (
        "complete" if report["valid_generation_count"] == 10 else "incomplete"
    )
    assert report["cross_view_action_identity"] == {"RacingCar": True}
    assert all(
        case["unity_reference"]["complete"] for case in report["cases"].values()
    )


def test_channel_coverage_and_declared_semantic_gaps() -> None:
    report = MODULE.build()
    racing = report["cases"]["table4/RacingCar/third"]
    billiard = report["cases"]["table4/BilliardGame/first"]

    assert racing["trajectory"]["channels"]["steer"]["missing_states"] == []
    assert racing["trajectory"]["channels"]["throttle"]["missing_states"] == [
        "negative"
    ]
    assert billiard["trajectory"]["channels"]["lateral"]["missing_states"] == []
    assert billiard["trajectory"]["channels"]["forward"]["missing_states"] == [
        "negative"
    ]
    assert billiard["evaluation_status"] == "diagnostic_only_camera_pose_leakage_risk"
    assert "table4/BilliardGame/third" not in report["cases"]


def test_reference_timeline_uses_compact_package_video(
    tmp_path: Path, monkeypatch
) -> None:
    capture = tmp_path / "capture"
    frames = capture / "frames"
    frames.mkdir(parents=True)
    (frames / "frame_0000.png").write_bytes(b"first")
    video = capture / "unity_groundtruth.mp4"
    video.write_bytes(b"video")

    def fake_run(command, **kwargs):
        assert command[-1] == str(video)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {"streams": [{"nb_read_frames": "300", "duration": "10.000000"}]}
            ),
            stderr="",
        )

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    result = MODULE.reference_timeline(capture)
    assert result["source"] == "unity_groundtruth_mp4"
    assert result["frame_count"] == 300
    assert result["duration_sec"] == 10.0
    assert result["video_sha256"] == MODULE.sha256(video)


def test_reference_timeline_prefers_complete_png_sequence(
    tmp_path: Path, monkeypatch
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(300):
        (frames / f"frame_{index:04d}.png").touch()

    def unexpected_run(*args, **kwargs):
        raise AssertionError("ffprobe should not run for a complete sequence")

    monkeypatch.setattr(MODULE.subprocess, "run", unexpected_run)
    result = MODULE.reference_timeline(tmp_path)
    assert result == {
        "frame_count": 300,
        "duration_sec": 10.0,
        "source": "png_sequence",
    }


def test_reference_timeline_uses_evaluation_artifacts_without_video(
    tmp_path: Path,
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_0000.png").touch()
    artifacts = tmp_path / "evaluation_artifacts"
    artifacts.mkdir()
    transforms = artifacts / "object_transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "objects": {
                    "car": {"frames": list(range(300))},
                    "road": {"frames": list(range(300))},
                },
            }
        ),
        encoding="utf-8",
    )
    result = MODULE.reference_timeline(tmp_path)
    assert result["source"] == "evaluation_artifact_timeline"
    assert result["frame_count"] == 300
    assert result["duration_sec"] == 10.0
    assert result["object_transforms_sha256"] == MODULE.sha256(transforms)
