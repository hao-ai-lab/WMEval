from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "prepare_project_genie_case_eval.py"
SPEC = importlib.util.spec_from_file_location("prepare_project_genie_case_eval", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_verified_artifact_prefers_aligned_native(tmp_path: Path) -> None:
    video = tmp_path / "trajectory_native.mp4"
    video.write_bytes(b"video")
    receipt = {
        "evaluation_artifact_status": "native_action_aligned_ready",
        "evaluation_video": str(video),
        "evaluation_video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "observed_video": str(tmp_path / "missing.mp4"),
    }
    assert MODULE.verified_artifact(receipt) == (video.resolve(), "native_aligned")


def test_verified_artifact_accepts_action_capture(tmp_path: Path) -> None:
    video = tmp_path / "observed.mp4"
    video.write_bytes(b"capture")
    receipt = {
        "evaluation_artifact_status": "action_aligned_capture_ready",
        "observed_video": str(video),
        "observed_video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
    }
    assert MODULE.verified_artifact(receipt) == (video.resolve(), "browser_capture")


def test_select_common_artifacts_does_not_mix_native_and_browser(tmp_path: Path) -> None:
    receipts = []
    for index in range(2):
        observed = tmp_path / f"observed_{index}.mp4"
        observed.write_bytes(f"capture-{index}".encode())
        receipt = {
            "evaluation_artifact_status": (
                "native_action_aligned_ready" if index == 0 else "action_aligned_capture_ready"
            ),
            "observed_video": str(observed),
            "observed_video_sha256": hashlib.sha256(observed.read_bytes()).hexdigest(),
        }
        if index == 0:
            native = tmp_path / "native.mp4"
            native.write_bytes(b"native")
            receipt.update(
                evaluation_video=str(native),
                evaluation_video_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
            )
        receipts.append(receipt)

    selected, policy = MODULE.select_common_artifacts(receipts)
    assert policy == "all_browser_capture_due_to_incomplete_native_coverage"
    assert [kind for _, kind in selected] == ["browser_capture", "browser_capture"]
    assert [path.name for path, _ in selected] == ["observed_0.mp4", "observed_1.mp4"]


def test_pose_yaw_mapping_follows_requested_direction_and_wraps() -> None:
    angles = np.deg2rad([0.0, -90.0, -180.0, -270.0, -360.0])
    poses = np.stack([np.eye(4) for _ in angles])
    poses[:, 0, 2] = np.sin(angles)
    poses[:, 2, 2] = np.cos(angles)
    assert MODULE.pose_yaw_source_indices(
        poses, start_frame=0, yaw_offsets_deg=np.array([0.0, -90.0, -180.0, -270.0])
    ) == [0, 1, 2, 3]
    assert MODULE.pose_yaw_source_indices(
        poses, start_frame=2, yaw_offsets_deg=np.array([0.0, 90.0, 180.0, 270.0])
    ) == [2, 1, 0, 3]


def test_media_tool_uses_repo_pinned_fallback(tmp_path: Path, monkeypatch) -> None:
    bundled = tmp_path / ".local" / "bin" / "ffmpeg"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"binary")
    monkeypatch.setattr(MODULE.shutil, "which", lambda _name: None)
    monkeypatch.setattr(MODULE, "REPO_ROOT", tmp_path)

    assert MODULE.media_tool("ffmpeg") == str(bundled)
