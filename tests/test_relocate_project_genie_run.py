from __future__ import annotations

import json
from pathlib import Path

from evaluation.project_genie_alignment import sha256
from tools.relocate_project_genie_run import relocate_run


def test_relocate_run_rewrites_paths_and_refreshes_hashes(tmp_path: Path) -> None:
    source = tmp_path / "run0"
    destination = tmp_path / "run0_calibration_revisit"
    source.mkdir()
    for name in ("native.mp4", "trajectory_native.mp4", "action_trace.json"):
        (source / name).write_bytes(name.encode())
    screenshot = source / "complete.png"
    screenshot.write_bytes(b"png")
    observed = source / "observed_frames.json"
    observed.write_text(json.dumps({"path": str(source / "frame.jpg")}))
    (source / "frame.jpg").write_bytes(b"frame")
    report_path = source / "native_alignment.json"
    report = {
        "native_video": str(source / "native.mp4"),
        "native_video_sha256": sha256(source / "native.mp4"),
        "observed_manifest": str(observed),
        "observed_manifest_sha256": sha256(observed),
        "action_trace": str(source / "action_trace.json"),
        "action_trace_sha256": sha256(source / "action_trace.json"),
        "trajectory_native": str(source / "trajectory_native.mp4"),
        "trajectory_native_sha256": sha256(source / "trajectory_native.mp4"),
    }
    report_path.write_text(json.dumps(report))
    receipt = {
        "native_video": str(source / "native.mp4"),
        "native_video_sha256": sha256(source / "native.mp4"),
        "observed_frames": str(observed),
        "observed_frames_sha256": sha256(observed),
        "action_trace": str(source / "action_trace.json"),
        "action_trace_sha256": sha256(source / "action_trace.json"),
        "evaluation_video": str(source / "trajectory_native.mp4"),
        "evaluation_video_sha256": sha256(source / "trajectory_native.mp4"),
        "native_alignment_report": str(report_path),
        "native_alignment_report_sha256": sha256(report_path),
        "screenshots": {
            "complete": {"path": str(screenshot), "sha256": sha256(screenshot)}
        },
    }
    (source / "receipt.json").write_text(json.dumps(receipt))

    updated = relocate_run(source, destination)

    assert not source.exists()
    assert destination.is_dir()
    assert updated["table_repeat_eligible"] is False
    assert updated["archive_reason"] == "calibration_only_reused_world"
    assert updated["observed_frames_sha256"] == sha256(
        destination / "observed_frames.json"
    )
    assert updated["native_alignment_report_sha256"] == sha256(
        destination / "native_alignment.json"
    )
    for path in destination.rglob("*.json"):
        assert f"{source}/" not in path.read_text(encoding="utf-8")
